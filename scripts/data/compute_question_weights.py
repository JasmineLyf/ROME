"""Export training-only priors from Equations (5) and (6)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from common import DATA_DIR, QUESTION_COLS, load_splits, load_metadata, load_answers, compute_priors


def main():
    train, _, _ = load_splits()
    importance, reliability = compute_priors(load_answers(), load_metadata(), train)
    output = DATA_DIR / 'priors'
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'question': QUESTION_COLS, 'q_importance': importance}).to_csv(output / 'q_importance.csv', index=False)
    pd.DataFrame({'question': QUESTION_COLS, 'q_uncertainty': 1 - reliability,
                  'q_reliability': reliability}).to_csv(output / 'q_reliability.csv', index=False)
    print(f'Saved training-only priors to {output}')


if __name__ == '__main__':
    main()
