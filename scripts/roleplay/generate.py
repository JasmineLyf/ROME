"""Offline Ask generation; held-out test users are never sent to the LLM."""
import argparse
import csv
from pathlib import Path
import re
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR, load_splits, load_metadata
from roleplay.mbti_questionnaire import load_questions, build_prompt

MODEL = 'gpt-4o-2024-08-06'
# Retain the five sampling temperatures found in the bundled Ask responses.
TEMPERATURES = [0.2, 0.3, 0.4, 0.5, 0.6]


def parse_answers(output):
    answers = [None] * 60
    predicted_type = ''
    for line in output.strip().splitlines():
        match = re.fullmatch(r'Q(\d+)\s*:\s*([+-]?[0-3])', line.strip(), re.IGNORECASE)
        if match:
            number, score = int(match[1]), int(match[2])
            if not 1 <= number <= 60 or answers[number - 1] is not None:
                raise ValueError('Invalid or duplicate question number.')
            answers[number - 1] = score
        elif re.fullmatch(r'[IE][SN][TF][PJ]', line.strip().upper()):
            predicted_type = line.strip().upper()
    if any(value is None for value in answers):
        raise ValueError('Expected 60 integer answers in [-3, 3]; incomplete output was not saved.')
    return predicted_type, answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DATA_DIR / 'roleplay/answers_60_gpt4o.csv')
    parser.add_argument('--model', default=MODEL)
    args = parser.parse_args()
    from openai import OpenAI
    client = OpenAI()
    questions = load_questions(DATA_DIR / 'questionnaire/mbti_questions.txt')
    if len(questions) != 60:
        raise ValueError('Expected 60 questionnaire items.')
    train, val, _ = load_splits()
    # Validation responses only select the Answer checkpoint; they receive no gradients.
    metadata = load_metadata().loc[train + val]
    fields = ['user_id', 'true_type', 'pred_type', 'temperature'] + [f'Q{i}' for i in range(1, 61)] + ['uids']
    existing = set()
    if args.output.exists() and args.output.stat().st_size:
        with args.output.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fields:
                raise ValueError('Existing CSV schema differs; select a separate --output file.')
            for row in reader:
                try:
                    values = [float(row[f'Q{i}']) for i in range(1, 61)]
                    valid = all(-3 <= x <= 3 and x.is_integer() for x in values)
                except (ValueError, TypeError):
                    valid = False
                if valid:
                    existing.add(row['user_id'])
                else:
                    existing.discard(row['user_id'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    failures = []
    with args.output.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for uid, user in metadata.iterrows():
            prompt = build_prompt(user['type'], user['posts'], questions)
            for temperature in TEMPERATURES:
                tag = f'{uid}_temp{temperature:.1f}'
                if tag in existing:
                    continue
                try:
                    response = client.chat.completions.create(
                        model=args.model, messages=[{'role': 'user', 'content': prompt}],
                        temperature=temperature,
                    )
                    predicted_type, answers = parse_answers(response.choices[0].message.content)
                except Exception as error:
                    failures.append(tag)
                    print(f'Failed {tag}: {error}')
                    continue
                row = dict(zip([f'Q{i}' for i in range(1, 61)], answers))
                row.update(user_id=tag, true_type=user['type'], pred_type=predicted_type,
                           temperature=temperature, uids=uid)
                writer.writerow(row)
                handle.flush()
                existing.add(tag)
                time.sleep(0.5)
    if failures:
        raise RuntimeError(f'{len(failures)} Ask trials failed; rerun to retry: {failures[:10]}')
    print(f'Saved Ask responses to {args.output}')


if __name__ == '__main__':
    main()
