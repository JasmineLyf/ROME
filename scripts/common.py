"""Shared data protocol for the Ask, Answer, and Detect stages."""
from pathlib import Path
import random
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
QUESTION_COLS = [f'Q{i}' for i in range(1, 61)]
DIMS = ['IE', 'SN', 'TF', 'PJ']
ASSIGNMENTS = (
    'IE SN TF PJ TF IE PJ TF PJ TF '
    'IE SN TF PJ TF IE SN TF SN TF '
    'IE SN TF PJ TF IE TF TF PJ TF '
    'IE SN TF PJ PJ IE SN TF PJ TF '
    'IE SN IE PJ TF SN TF TF PJ TF '
    'IE SN IE TF TF PJ SN TF PJ TF'
).split()


def get_question2dim():
    return dict(zip(QUESTION_COLS, ASSIGNMENTS))


def question_onehot():
    return np.eye(4, dtype=np.float32)[[DIMS.index(d) for d in ASSIGNMENTS]]


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_uids(path):
    ids = [int(x) for x in Path(path).read_text().split()]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate user IDs in {path}')
    return ids


def load_splits(split_dir=DATA_DIR / 'splits'):
    """Use explicit 60/20/20 files, or subdivide the bundled 80% training pool."""
    split_dir = Path(split_dir)
    train = load_uids(split_dir / 'train_uids.txt')
    test = load_uids(split_dir / 'test_uids.txt')
    val_path = split_dir / 'val_uids.txt'
    if val_path.exists():
        val = load_uids(val_path)
    else:
        train, val = train_test_split(train, test_size=0.25, random_state=42, shuffle=True)
    parts = [set(train), set(val), set(test)]
    if any(not part for part in parts) or any(parts[i] & parts[j] for i in range(3) for j in range(i)):
        raise ValueError('Train, validation, and test users must be nonempty and disjoint.')
    total = sum(map(len, parts))
    if any(abs(len(part) - total * ratio) > 2 for part, ratio in zip(parts, [0.6, 0.2, 0.2])):
        raise ValueError('Expected user-level 60/20/20 splits; rerun scripts/data/datasplit.py.')
    return tuple(sorted(part) for part in parts)


def load_metadata(path=DATA_DIR / 'mbti_1.csv'):
    df = pd.read_csv(path)
    if 'uids' not in df:
        df['uids'] = np.arange(len(df))
    numeric_ids = pd.to_numeric(df['uids'], errors='raise')
    if numeric_ids.isna().any() or (numeric_ids % 1 != 0).any():
        raise ValueError('User IDs must be finite integers.')
    df['uids'] = numeric_ids.astype(int)
    if df.uids.duplicated().any():
        raise ValueError('Metadata contains duplicate user IDs.')
    if not df['type'].str.fullmatch(r'[IE][SN][TF][PJ]').all():
        raise ValueError('Invalid MBTI labels.')
    return df.set_index('uids', drop=False)


def load_answers(path=DATA_DIR / 'roleplay/answers_60_gpt4o.csv'):
    df = pd.read_csv(path, low_memory=False)
    if 'user_id' in df:
        # A successful retry supersedes an earlier incomplete sample with the same tag.
        df = df.drop_duplicates('user_id', keep='last').copy()
    if 'uids' not in df:
        df['uids'] = df['user_id'].astype(str).str.extract(r'^(\d+)_', expand=False)
    numeric_ids = pd.to_numeric(df['uids'], errors='raise')
    if numeric_ids.isna().any() or (numeric_ids % 1 != 0).any():
        raise ValueError('User IDs must be finite integers.')
    df['uids'] = numeric_ids.astype(int)
    df[QUESTION_COLS] = df[QUESTION_COLS].apply(pd.to_numeric, errors='coerce')
    values = df[QUESTION_COLS].to_numpy()
    finite = values[np.isfinite(values)]
    if np.isinf(values).any() or ((finite < -3) | (finite > 3)).any():
        raise ValueError('Answers must use the paper\'s [-3, 3] Likert scale.')
    return df


def answer_targets(answers, uids):
    selected = answers[answers.uids.isin(uids)]
    missing = int(selected[QUESTION_COLS].isna().sum().sum())
    if missing:
        warnings.warn(
            f'{missing} missing Ask scores in selected users; means use available samples. '
            'Repair incomplete trials before claiming fixed-T reproduction.', RuntimeWarning, stacklevel=2)
    targets = selected.groupby('uids')[QUESTION_COLS].mean().reindex(uids)
    if not np.isfinite(targets.to_numpy()).all():
        raise ValueError('Missing question targets for selected users; regenerate incomplete Ask samples.')
    return targets


def minmax_norm(values):
    values = np.asarray(values, dtype=np.float64)
    span = np.ptp(values)
    return np.zeros_like(values) if span == 0 else (values - values.min()) / span


def compute_priors(answers, metadata, train_uids):
    """Equations (5)-(7); never use validation/test labels or responses."""
    samples = answers[answers.uids.isin(train_uids)]
    means = answer_targets(samples, train_uids)
    grouped = samples.groupby('uids')[QUESTION_COLS]
    if (grouped.count().reindex(train_uids).fillna(0).to_numpy() < 2).any():
        raise ValueError('Reliability needs at least two valid samples per training user and question.')
    uncertainty = grouped.var(ddof=0).mean().reindex(QUESTION_COLS).to_numpy()
    labels = metadata.loc[means.index, 'type']
    importance = []
    for q, dim in get_question2dim().items():
        pos = labels.str[DIMS.index(dim)] == dim[1]
        if not pos.any() or pos.all():
            raise ValueError(f'Both classes are required to estimate importance for {dim}.')
        importance.append(abs(means.loc[pos, q].mean() - means.loc[~pos, q].mean()))
    return minmax_norm(importance).astype(np.float32), (1 - minmax_norm(uncertainty)).astype(np.float32)


def load_user_embeddings(emb_dir=DATA_DIR / 'embeddings'):
    emb_dir = Path(emb_dir)
    totals, counts, seen_posts = {}, {}, set()
    embedding_dim = None
    for split in ['train', 'val', 'test']:
        path = emb_dir / f'{split}_post_embeddings.npy'
        if split == 'val' and not path.exists():
            continue
        arr = np.load(path, mmap_mode='r')
        index = np.load(emb_dir / f'{split}_post_index_map.npy', allow_pickle=True)
        if arr.ndim != 2 or index.shape != (len(arr), 2):
            raise ValueError(f'Invalid embedding/index shapes for {split}.')
        if embedding_dim is not None and arr.shape[1] != embedding_dim:
            raise ValueError('All post embeddings must have the same feature dimension.')
        embedding_dim = arr.shape[1]
        if len(arr) != len(index):
            raise ValueError(f'Embedding/index length mismatch for {split}.')
        for (uid, post_id), emb in zip(index, arr):
            if int(uid) != uid or int(post_id) != post_id:
                raise ValueError('Embedding index IDs must be integers.')
            uid, post_id = int(uid), int(post_id)
            if (uid, post_id) in seen_posts:
                raise ValueError('Duplicate post across embedding files; regenerate all splits together.')
            seen_posts.add((uid, post_id))
            if not np.isfinite(emb).all():
                raise ValueError(f'Nonfinite embedding for user {uid}.')
            if uid not in totals:
                totals[uid] = np.zeros(emb.shape, dtype=np.float32)
                counts[uid] = 0
            totals[uid] += emb
            counts[uid] += 1
    return {uid: total / counts[uid] for uid, total in totals.items()}


def macro_f1(y_true, y_pred):
    scores = [f1_score(y_true[:, i], y_pred[:, i], average='macro', labels=[0, 1], zero_division=0)
              for i in range(4)]
    return scores, float(np.mean(scores))


def validate_split_coverage(metadata, partitions):
    assigned = set().union(*(set(ids) for ids in partitions))
    if assigned != set(metadata.index):
        raise ValueError('Split users must cover the metadata exactly; regenerate or restore matching splits.')
