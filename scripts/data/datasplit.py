# datasplit.py
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Add a sequential uids column to the original dataset and write train/test uid splits."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/mbti_1.csv",
        help="Path to the raw CSV (relative to the ROME project root).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/splits",
        help="Output directory for split files (relative to the ROME project root).",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Disable shuffling (default: shuffle=True).",
    )
    args = parser.parse_args()

    shuffle = not args.no_shuffle

    # Load dataset
    df = pd.read_csv(args.input_csv)

    # Add sequential uids (top-to-bottom order) and write back to the same CSV
    df["uids"] = range(len(df))
    cols = ["uids"] + [c for c in df.columns if c != "uids"]
    df = df[cols]
    df.to_csv(args.input_csv, index=False)

    # Split uids only
    uids = df["uids"].tolist()
    train_uids, test_uids = train_test_split(
        uids,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=shuffle,
    )

    # Sort for stable, diff-friendly outputs
    train_uids = sorted(train_uids)
    test_uids = sorted(test_uids)

    # Write split files
    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train_uids.txt")
    test_path = os.path.join(args.out_dir, "test_uids.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, train_uids)) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, test_uids)) + "\n")

    print(f"[OK] Updated dataset in-place: {args.input_csv} (added 'uids' column).")
    print(f"[OK] Total: {len(uids)} | Train: {len(train_uids)} | Test: {len(test_uids)}")
    print(f"[OK] Wrote: {train_path}")
    print(f"[OK] Wrote: {test_path}")


if __name__ == "__main__":
    main()
