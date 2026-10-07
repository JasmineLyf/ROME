"""Check source files and locally generated training inputs.

A fresh source checkout does not include generated artifacts. Use --strict after
preprocessing to require complete training inputs. This is not a score comparison.
"""
import argparse
import json
import warnings
from pathlib import Path
from common import (DATA_DIR, QUESTION_COLS, load_metadata, load_answers,
                    load_splits, answer_targets, compute_priors, validate_split_coverage)
from data.export_embeddings import load_questions


def audit(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    report = {'source_ready': False, 'training_ready': False, 'next_steps': [], 'errors': []}
    try:
        metadata = load_metadata(data_dir / 'mbti_1.csv')
        load_questions(data_dir / 'questionnaire/mbti_questions.txt')
        report['source_ready'] = True
    except (OSError, ValueError, KeyError) as error:
        report['errors'].append(str(error))
        return report
    split_dir = data_dir / 'splits'
    if not all((split_dir / name).is_file() for name in ['train_uids.txt', 'test_uids.txt']):
        report['next_steps'].append('python scripts/data/datasplit.py')
        partitions = None
    else:
        try:
            partitions = load_splits(split_dir)
            validate_split_coverage(metadata, partitions)
            report['split_sizes'] = dict(zip(['train', 'val', 'test'], map(len, partitions)))
        except (OSError, ValueError, KeyError) as error:
            report['errors'].append(str(error))
            partitions = None
    answer_path = data_dir / 'roleplay/answers_60_gpt4o.csv'
    if not answer_path.is_file():
        report['next_steps'].append('Set OPENAI_API_KEY, then run: python scripts/roleplay/generate.py')
    elif partitions is not None:
        try:
            answers = load_answers(answer_path)
            selected = answers[answers.uids.isin(partitions[0] + partitions[1])]
            if selected[QUESTION_COLS].isna().any().any():
                report['errors'].append('Incomplete Ask trials; rerun scripts/roleplay/generate.py to repair them.')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                answer_targets(answers, partitions[0] + partitions[1])
                compute_priors(answers, metadata, partitions[0])
        except (OSError, ValueError, KeyError) as error:
            report['errors'].append(str(error))
    embedding_names = ['question_embeddings.npy']
    for split in ['train', 'val', 'test']:
        embedding_names.extend([f'{split}_post_embeddings.npy', f'{split}_post_index_map.npy'])
    if any(not (data_dir / 'embeddings' / name).is_file() for name in embedding_names):
        report['next_steps'].append('python scripts/data/export_embeddings.py')
    report['prior_export'] = 'Optional: Detect computes priors directly from training users.'
    report['training_ready'] = not report['next_steps'] and not report['errors']
    report['verification_scope'] = 'Checks data prerequisites; embedding numerical checks also run in training.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    report = audit(args.data_dir)
    print(json.dumps(report, indent=2))
    if report['errors'] or (args.strict and not report['training_ready']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
