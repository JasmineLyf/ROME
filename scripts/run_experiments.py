"""Repeat both training stages over five seeds and report mean/std Macro-F1."""
import argparse
import json
import numpy as np
from common import ROOT, DIMS
from train_answer_pretrain import train_answer_pretrain
from train_detect import train_detect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--answer-epochs', type=int, default=120)
    parser.add_argument('--detect-epochs', type=int, default=50)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be distinct.')
    if args.answer_epochs < 1 or args.detect_epochs < 1:
        parser.error('Epoch counts must be positive.')
    runs = []
    for seed in args.seeds:
        output = ROOT / 'checkpoints' / f'seed_{seed}'
        pretrain = train_answer_pretrain(seed, args.answer_epochs, output / 'best_pretrain.pth')
        runs.append(train_detect(seed, args.detect_epochs, pretrain, output / 'best_detect.pth', output / 'metrics.json'))
    values = np.array([[r['per_dimension_macro_f1'][d] for d in DIMS] + [r['average_macro_f1']] for r in runs])
    result = {'runs': runs, 'mean': dict(zip(DIMS + ['average'], values.mean(axis=0).tolist())),
              'std': dict(zip(DIMS + ['average'], values.std(axis=0, ddof=0).tolist())), 'std_ddof': 0}
    target = ROOT / 'checkpoints' / 'five_seed_metrics.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
