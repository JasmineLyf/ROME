"""Create disjoint user-level 60/20/20 partitions without changing raw data."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.model_selection import train_test_split
from common import DATA_DIR, load_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=Path, default=DATA_DIR / 'mbti_1.csv')
    parser.add_argument('--out_dir', type=Path, default=DATA_DIR / 'splits')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    users = load_metadata(args.input_csv).index.tolist()
    pool, test = train_test_split(users, test_size=0.2, random_state=args.seed, shuffle=True)
    train, val = train_test_split(sorted(pool), test_size=0.25, random_state=args.seed, shuffle=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in [('train', train), ('val', val), ('test', test)]:
        (args.out_dir / f'{name}_uids.txt').write_text('\n'.join(map(str, sorted(ids))) + '\n')
        print(f'{name}: {len(ids)} users')


if __name__ == '__main__':
    main()
