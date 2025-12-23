import csv
import time
import re
import os

from openai import OpenAI
from src.roleplay.mbti_questionnaire import load_questions, build_prompt

# client = OpenAI(api_key="YOUR_API_KEY")  # Do NOT hardcode keys
client = OpenAI()

MODEL = "gpt-4o"
TEMPERATURE = 0.7
N_TRIALS = 5
SLEEP_SECONDS = 0.5


def load_user_data(csv_path):
    user_data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            mbti_type = (row.get("type") or "").strip()
            posts = (row.get("posts") or "").strip()
            if not mbti_type or not posts:
                continue

            uid_raw = row.get("uids", "").strip()
            uid = int(uid_raw) if uid_raw.isdigit() else idx
            user_data.append((uid, mbti_type, posts))
    return user_data


def call_gpt(prompt):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    return resp.choices[0].message.content


def parse_answers(output):
    lines = output.strip().splitlines()
    answers = [None] * 60
    pred_type = ""

    for line in lines:
        if line.strip().lower().startswith("q") and ":" in line:
            try:
                qidx, score = line.strip().split(":", 1)
                qnum = int(qidx.lower().replace("q", "").strip()) - 1
                if 0 <= qnum < 60:
                    answers[qnum] = score.strip()
            except Exception:
                pass

    for line in reversed(lines):
        m = re.fullmatch(r"[EINPSFTJ]{4}", line.strip().upper())
        if m:
            pred_type = m.group(0)
            break

    return pred_type, answers


def load_existing_tags(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row["user_id"])
    return done


def ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    questions_path = "data/questionnaire/mbti_questions.txt"
    raw_csv_path = "data/mbti_1.csv"
    output_file = "data/roleplay/answers_60_gpt4o.csv"

    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Missing questions file: {questions_path}")
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Missing raw data file: {raw_csv_path}")

    ensure_parent_dir(output_file)

    questions = load_questions(questions_path)
    user_data = load_user_data(raw_csv_path)

    existing = load_existing_tags(output_file)
    write_header = (not os.path.exists(output_file)) or (os.stat(output_file).st_size == 0)

    fieldnames = ["user_id", "true_type", "pred_type"] + [f"Q{i+1}" for i in range(60)]

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for uid, true_type, posts in user_data:
            for trial in range(N_TRIALS):
                tag = f"{uid}_gpt4o_trial{trial+1}"
                if tag in existing:
                    continue

                prompt = build_prompt(true_type, posts, questions)

                try:
                    resp = call_gpt(prompt)
                    pred_type, answers = parse_answers(resp)
                except Exception as e:
                    print(f"Error for {tag}: {e}")
                    continue

                row = {"user_id": tag, "true_type": true_type, "pred_type": pred_type}
                for i in range(60):
                    row[f"Q{i+1}"] = answers[i] if answers[i] is not None else ""
                writer.writerow(row)
                f.flush()

                time.sleep(SLEEP_SECONDS)

    print(f"[OK] Wrote: {output_file}")


if __name__ == "__main__":
    main()
