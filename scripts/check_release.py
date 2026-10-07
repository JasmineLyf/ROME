"""Audit bundled data against the paper without generating answers or training.

Usage: python scripts/check_release.py [--strict]
Strict mode fails when known data limitations prevent a verified paper reproduction.
"""
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from common import (DIMS, QUESTION_COLS, DATA_DIR, load_metadata, load_answers,
                    load_splits, answer_targets, compute_priors, validate_split_coverage)

PAPER_COUNTS = {
    'train': [[4032, 1173], [724, 4481], [2388, 2817], [3160, 2045]],
    'val': [[1330, 405], [230, 1505], [802, 933], [1007, 728]],
    'test': [[1314, 421], [243, 1492], [791, 944], [1074, 661]],
}


def audit():
    metadata, answers, partitions = load_metadata(), load_answers(), load_splits()
    validate_split_coverage(metadata, partitions)
    counts = {}
    for name, ids in zip(['train', 'val', 'test'], partitions):
        labels = metadata.loc[ids, 'type']
        counts[name] = [[int((labels.str[i] == c).sum()) for c in dim]
                        for i, dim in enumerate(DIMS)]
    selected = answers[answers.uids.isin(partitions[0] + partitions[1])]
    incomplete = selected[selected[QUESTION_COLS].isna().any(axis=1)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        answer_targets(answers, partitions[0] + partitions[1])
        importance, reliability = compute_priors(answers, metadata, partitions[0])
    limitations = []
    if counts != PAPER_COUNTS:
        limitations.append('Bundled user IDs do not reproduce Table 4 class counts; original experiment splits are needed.')
    if not incomplete.empty:
        limitations.append('Incomplete Ask trials must be regenerated for fixed-T reproduction.')
    saved_importance = pd.read_csv(DATA_DIR / 'priors/q_importance.csv').set_index('question')
    saved_reliability = pd.read_csv(DATA_DIR / 'priors/q_reliability.csv').set_index('question')
    old_rel = (saved_reliability['q_reliability'] if 'q_reliability' in saved_reliability
               else 1 - saved_reliability['q_uncertainty'])
    stale_priors = not (np.allclose(saved_importance.loc[QUESTION_COLS, 'q_importance'], importance, atol=1e-6)
                       and np.allclose(old_rel.loc[QUESTION_COLS], reliability, atol=1e-6))
    return {
        'split_sizes': dict(zip(['train', 'val', 'test'], map(len, partitions))),
        'dimensions': DIMS,
        'actual_class_counts': counts,
        'paper_table4_class_counts': PAPER_COUNTS,
        'incomplete_training_or_validation_trials': incomplete[['user_id', 'uids']].to_dict('records'),
        'missing_training_or_validation_scores': int(selected[QUESTION_COLS].isna().sum().sum()),
        'bundled_priors_are_stale': stale_priors,
        'prior_handling': 'Detect recomputes priors from training users; exported legacy priors are not used.',
        'reproduction_limitations': limitations,
        'scope': 'Kaggle main pipeline only; Pandora, baselines, and ablation runners are not included.',
        'verification_limit': 'This audit does not verify paper scores or the provenance of cached LLM responses.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    report = audit()
    print(json.dumps(report, indent=2))
    if args.strict and report['reproduction_limitations']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
