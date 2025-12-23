import os
import re
import json
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_uids(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines()]
    return [int(x) for x in lines if x]


def load_questions(txt_path):
    pat = re.compile(r"^\s*Q(\d+)\s*:\s*(.+?)\s*$")
    q_map = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pat.match(line)
            if not m:
                continue
            qid = int(m.group(1))
            qtext = m.group(2).strip()
            if qtext:
                q_map[qid] = qtext

    questions = [q_map[i] for i in range(1, 61) if i in q_map]
    if len(questions) != 60:
        raise ValueError(f"Expected 60 questions (Q1..Q60), got {len(questions)}.")
    return questions


def get_embeddings(texts, tokenizer, model, device, batch_size=64, max_length=256):
    embeddings = []
    model.to(device)
    model.eval()

    total = len(texts)
    print(f"Computing embeddings for {total} texts (batch_size={batch_size}) on {device}")

    with torch.no_grad():
        for i in tqdm(range(0, total, batch_size), desc="Embedding batches", unit="batch"):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)

    if not embeddings:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.vstack(embeddings)


def build_posts_for_uids(df, uids, posts_col="posts", post_delim="|||"):
    all_posts = []
    index_map = []

    # df is indexed by uids (int)
    for uid in uids:
        if uid not in df.index:
            continue

        row = df.loc[uid]
        posts_field = row.get(posts_col, "")

        if not isinstance(posts_field, str) or not posts_field.strip():
            continue

        posts = posts_field.split(post_delim)
        for post_idx, post in enumerate(posts):
            text = str(post).strip().strip('"').strip("'")
            if not text:
                continue
            all_posts.append(text)
            index_map.append((int(uid), int(post_idx)))

    return all_posts, index_map


if __name__ == "__main__":
    # ---- Paths (match your new repo layout) ----
    RAW_CSV = "data/mbti_1.csv"
    SPLIT_DIR = "data/splits"  # train_uids.txt, test_uids.txt
    OUT_DIR = "data/embeddings"
    QUESTIONS_TXT = "data/questionnaire/mbti_questions.txt"

    # ---- Model / embed params ----
    MODEL_NAME = "bert-base-uncased"
    BATCH_SIZE = 64
    MAX_LENGTH = 256

    # ---- Output files ----
    os.makedirs(OUT_DIR, exist_ok=True)

    train_ids_path = os.path.join(SPLIT_DIR, "train_uids.txt")
    test_ids_path  = os.path.join(SPLIT_DIR, "test_uids.txt")

    for p in [RAW_CSV, train_ids_path, test_ids_path, QUESTIONS_TXT]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    device = select_device()
    print(f"Selected device: {device}")

    # ---- Load data and enforce uids semantics (top-to-bottom index) ----
    df = pd.read_csv(RAW_CSV)
    if "uids" not in df.columns:
        df.insert(0, "uids", range(len(df)))
    else:
        df["uids"] = pd.to_numeric(df["uids"], errors="coerce").astype(int)

    df = df.set_index("uids", drop=False)

    # ---- Load model ----
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)

    # ---- Process train/test splits ----
    split_files = {"train": train_ids_path, "test": test_ids_path}
    stats = {}

    for split, ids_path in split_files.items():
        uids = load_uids(ids_path)
        all_posts, index_map = build_posts_for_uids(df, uids)

        print(f"\nProcessing '{split}' split: {len(all_posts)} posts from {len(uids)} users (uids file)")

        post_emb = get_embeddings(
            all_posts,
            tokenizer,
            model,
            device,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        ).astype(np.float32)

        emb_path = os.path.join(OUT_DIR, f"{split}_post_embeddings.npy")
        map_path = os.path.join(OUT_DIR, f"{split}_post_index_map.npy")

        np.save(emb_path, post_emb)
        np.save(map_path, np.array(index_map, dtype=object))

        print(f"Saved: {emb_path}  shape={post_emb.shape}")
        print(f"Saved: {map_path}  size={len(index_map)}")

        stats[split] = {
            "uids_file": ids_path,
            "n_users_in_split_file": len(uids),
            "n_posts_used": len(all_posts),
            "emb_shape": list(post_emb.shape),
        }

    # ---- Question embeddings (Q1..Q60) ----
    questions = load_questions(QUESTIONS_TXT)
    q_emb = get_embeddings(
        questions,
        tokenizer,
        model,
        device,
        batch_size=64,
        max_length=MAX_LENGTH,
    ).astype(np.float32)

    q_path = os.path.join(OUT_DIR, "question_embeddings.npy")
    np.save(q_path, q_emb)
    print(f"\nSaved: {q_path}  shape={q_emb.shape}")

    # ---- Manifest ----
    manifest = {
        "raw_csv": RAW_CSV,
        "split_dir": SPLIT_DIR,
        "questions_txt": QUESTIONS_TXT,
        "model_name": MODEL_NAME,
        "pooling": "cls",
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "device": str(device),
        "outputs": {
            "dir": OUT_DIR,
            "files": [
                "train_post_embeddings.npy",
                "train_post_index_map.npy",
                "test_post_embeddings.npy",
                "test_post_index_map.npy",
                "questions_embeddings.npy",
                "manifest.json",
            ],
        },
        "stats": {
            **stats,
            "questions": {"n_questions": len(questions), "emb_shape": list(q_emb.shape)},
        },
    }

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Wrote manifest: {manifest_path}")
# build_embeddings.py
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_uids(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines()]
    return [int(x) for x in lines if x]


def load_questions(txt_path):
    # Expected format per line: "Q1: ...", "Q2: ...", ..., "Q60: ..."
    pat = re.compile(r"^\s*Q(\d+)\s*:\s*(.+?)\s*$")
    q_map = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pat.match(line)
            if not m:
                continue
            qid = int(m.group(1))
            qtext = m.group(2).strip()
            if qtext:
                q_map[qid] = qtext

    questions = [q_map[i] for i in range(1, 61) if i in q_map]
    if len(questions) != 60:
        raise ValueError(f"Expected 60 questions (Q1..Q60), got {len(questions)}.")
    return questions


def get_embeddings(texts, tokenizer, model, device, batch_size=64, max_length=256):
    # CLS pooling: take last_hidden_state[:, 0, :]
    embeddings = []
    model.to(device)
    model.eval()

    total = len(texts)
    print(f"Computing embeddings for {total} texts (batch_size={batch_size}) on {device}")

    with torch.no_grad():
        for i in tqdm(range(0, total, batch_size), desc="Embedding batches", unit="batch"):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)

    if not embeddings:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.vstack(embeddings)


def build_posts_for_split(df, uids, posts_col="posts", post_delim="|||"):
    # Collect all post texts and an index map: (uid, post_idx)
    all_posts = []
    index_map = []

    for uid in uids:
        if uid not in df.index:
            continue

        row = df.loc[uid]
        posts_field = row.get(posts_col, "")

        if not isinstance(posts_field, str) or not posts_field.strip():
            continue

        posts = posts_field.split(post_delim)
        for post_idx, post in enumerate(posts):
            text = str(post).strip().strip('"').strip("'")
            if not text:
                continue
            all_posts.append(text)
            index_map.append((int(uid), int(post_idx)))

    return all_posts, index_map


def ensure_uids_column_inplace(csv_path, uid_col="uids"):
    # Ensure the CSV has a 'uids' column in sequential order and write back in-place.
    df = pd.read_csv(csv_path)
    if uid_col not in df.columns:
        df[uid_col] = range(len(df))
    else:
        # Overwrite to guarantee 0..N-1 in top-to-bottom order (consistent with splits)
        df[uid_col] = range(len(df))

    # Move uids to the first column for readability (optional)
    cols = [uid_col] + [c for c in df.columns if c != uid_col]
    df = df[cols]
    df.to_csv(csv_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Compute BERT CLS embeddings for MBTI posts and questions.")
    parser.add_argument("--raw_csv", type=str, default="data/mbti_1.csv", help="Path to mbti_1.csv")
    parser.add_argument("--split_dir", type=str, default="data/splits", help="Directory containing split uid files")
    parser.add_argument("--out_dir", type=str, default="data/embeddings", help="Output directory for embeddings")
    parser.add_argument("--questions_txt", type=str, default="data/questionnaire/mbti_questions.txt",
                        help="Path to mbti_questions.txt (Q1..Q60)")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="HF model name")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--posts_col", type=str, default="posts", help="Column name for posts")
    parser.add_argument("--post_delim", type=str, default="|||", help="Delimiter used to split posts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = select_device()
    print(f"Selected device: {device}")

    if not os.path.exists(args.raw_csv):
        raise FileNotFoundError(f"Missing raw csv: {args.raw_csv}")

    train_ids_path = os.path.join(args.split_dir, "train_uids.txt")
    test_ids_path = os.path.join(args.split_dir, "test_uids.txt")

    for p in [train_ids_path, test_ids_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split file: {p}")

    if not os.path.exists(args.questions_txt):
        raise FileNotFoundError(f"Missing questions file: {args.questions_txt}")

    # Ensure 'uids' exists and is sequential; write back in-place to raw CSV
    df = ensure_uids_column_inplace(args.raw_csv, uid_col="uids")
    df = df.set_index("uids", drop=False)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name)

    split_files = {
        "train": train_ids_path,
        "test": test_ids_path,
    }

    stats = {}

    # Post embeddings for train/test
    for split, ids_path in split_files.items():
        uids = load_uids(ids_path)
        all_posts, index_map = build_posts_for_split(
            df,
            uids,
            posts_col=args.posts_col,
            post_delim=args.post_delim,
        )

        print(f"\nProcessing '{split}' split: {len(all_posts)} posts from {len(uids)} users (uids file)")

        post_emb = get_embeddings(
            all_posts,
            tokenizer,
            model,
            device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        ).astype(np.float32)

        emb_path = os.path.join(args.out_dir, f"{split}_post_embeddings.npy")
        map_path = os.path.join(args.out_dir, f"{split}_post_index_map.npy")

        np.save(emb_path, post_emb)
        np.save(map_path, np.array(index_map, dtype=object))

        print(f"Saved: {emb_path}  shape={post_emb.shape}")
        print(f"Saved: {map_path}  size={len(index_map)}")

        stats[split] = {
            "n_users_in_split_file": len(uids),
            "n_posts_used": len(all_posts),
            "emb_shape": list(post_emb.shape),
        }

    # Question embeddings
    questions = load_questions(args.questions_txt)
    q_emb = get_embeddings(
        questions,
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).astype(np.float32)

    q_path = os.path.join(args.out_dir, "questions_embeddings.npy")
    np.save(q_path, q_emb)
    print(f"\nSaved: {q_path}  shape={q_emb.shape}")

    # Manifest
    manifest = {
        "raw_csv": args.raw_csv,
        "split_dir": args.split_dir,
        "questions_txt": args.questions_txt,
        "model_name": args.model_name,
        "pooling": "cls",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": str(device),
        "outputs": {
            "dir": args.out_dir,
            "files": [
                "train_post_embeddings.npy",
                "train_post_index_map.npy",
                "test_post_embeddings.npy",
                "test_post_index_map.npy",
                "questions_embeddings.npy",
                "manifest.json",
            ],
        },
        "stats": {
            **stats,
            "questions": {"n_questions": len(questions), "emb_shape": list(q_emb.shape)},
        },
        "notes": {
            "uids_policy": "uids overwritten in-place as 0..N-1 in top-to-bottom order in raw_csv",
            "posts_col": args.posts_col,
            "post_delim": args.post_delim,
        },
    }

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
