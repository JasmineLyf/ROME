import os
import re
import numpy as np
import pandas as pd

EPS = 1e-8
DIM2IDX = {"IE": 0, "SN": 1, "TF": 2, "PJ": 3}

def get_question2dim():
    return {
        "Q1":"IE","Q2":"SN","Q3":"TF","Q4":"PJ","Q5":"TF","Q6":"IE","Q7":"PJ","Q8":"TF","Q9":"PJ","Q10":"TF",
        "Q11":"IE","Q12":"SN","Q13":"TF","Q14":"PJ","Q15":"TF","Q16":"IE","Q17":"SN","Q18":"TF","Q19":"SN","Q20":"TF",
        "Q21":"IE","Q22":"SN","Q23":"TF","Q24":"PJ","Q25":"TF","Q26":"IE","Q27":"TF","Q28":"TF","Q29":"PJ","Q30":"TF",
        "Q31":"IE","Q32":"SN","Q33":"TF","Q34":"PJ","Q35":"PJ","Q36":"IE","Q37":"SN","Q38":"TF","Q39":"PJ","Q40":"TF",
        "Q41":"IE","Q42":"SN","Q43":"IE","Q44":"PJ","Q45":"TF","Q46":"SN","Q47":"TF","Q48":"TF","Q49":"PJ","Q50":"TF",
        "Q51":"IE","Q52":"SN","Q53":"IE","Q54":"TF","Q55":"TF","Q56":"PJ","Q57":"SN","Q58":"TF","Q59":"PJ","Q60":"TF"
    }

def minmax_norm(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + EPS)

def load_uids(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]

def parse_uid_from_tag(tag):
    # Expected: "{uid}_gpt4o_trial{t}"
    m = re.match(r"^(\d+)_", str(tag))
    return int(m.group(1)) if m else None

def main():
    RAW_CSV = "data/mbti_1.csv"
    TRAIN_UIDS_TXT = "data/splits/train_uids.txt"
    ROLEPLAY_CSV = "data/roleplay/answers_60_gpt4o.csv"
    OUT_DIR = "data/priors"

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Missing: {RAW_CSV}")
    if not os.path.exists(TRAIN_UIDS_TXT):
        raise FileNotFoundError(f"Missing: {TRAIN_UIDS_TXT}")
    if not os.path.exists(ROLEPLAY_CSV):
        raise FileNotFoundError(f"Missing: {ROLEPLAY_CSV}")

    train_uids = set(load_uids(TRAIN_UIDS_TXT))

    df_raw = pd.read_csv(RAW_CSV)
    df_raw["uids"] = df_raw.index
    uid2type = dict(zip(df_raw["uids"].astype(int), df_raw["type"].astype(str)))

    df = pd.read_csv(ROLEPLAY_CSV)
    if "user_id" not in df.columns:
        raise ValueError("ROLEPLAY_CSV must contain column: user_id")

    df["uids"] = df["user_id"].apply(parse_uid_from_tag)
    df = df.dropna(subset=["uids"]).copy()
    df["uids"] = df["uids"].astype(int)
    df = df[df["uids"].isin(train_uids)].copy()
    if df.empty:
        raise ValueError("No roleplay rows matched train_uids. Check splits and roleplay output.")

    answer_cols = [f"Q{i+1}" for i in range(60)]
    for c in answer_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in ROLEPLAY_CSV: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["mbti_type"] = df["uids"].map(uid2type)
    df = df.dropna(subset=["mbti_type"]).copy()
    df["mbti_type"] = df["mbti_type"].astype(str)

    long_ans = (
        df.melt(id_vars=["uids"], value_vars=answer_cols, var_name="question", value_name="response")
          .dropna(subset=["response"])
    )

    # Reliability: q_unc -> minmax -> q_rel = 1 - norm(q_unc)
    ans_var = (
        long_ans.groupby(["uids", "question"])["response"]
               .var(ddof=0)
               .reset_index(name="answer_var")
    )
    q_unc = (
        ans_var.groupby("question")["answer_var"]
               .mean()
               .reset_index(name="q_unc_raw")
    )
    q_rel = q_unc.copy()
    q_rel["q_rel_norm"] = minmax_norm(q_rel["q_unc_raw"].values)
    q_rel["q_reliability"] = 1.0 - q_rel["q_rel_norm"]
    q_rel_out = q_rel[["question", "q_reliability"]].sort_values("question").reset_index(drop=True)

    # Importance: q_imp = |mu_pos - mu_neg| on per-user mean across trials, then minmax
    ans_mean = (
        long_ans.groupby(["uids", "question"])["response"]
               .mean()
               .reset_index(name="a_tilde")
    )
    ans_mean["mbti_type"] = ans_mean["uids"].map(uid2type).astype(str)

    q2dim = get_question2dim()
    imp_records = []

    for q in answer_cols:
        dim = q2dim[q]
        idx = DIM2IDX[dim]
        l_pos, l_neg = dim[0], dim[1]

        sub = ans_mean[ans_mean["question"] == q]
        grp_pos = sub[sub["mbti_type"].str[idx] == l_pos]["a_tilde"]
        grp_neg = sub[sub["mbti_type"].str[idx] == l_neg]["a_tilde"]

        mu_pos = float(grp_pos.mean()) if len(grp_pos) > 0 else 0.0
        mu_neg = float(grp_neg.mean()) if len(grp_neg) > 0 else 0.0

        imp_records.append({"question": q, "q_imp_raw": abs(mu_pos - mu_neg)})

    q_imp = pd.DataFrame(imp_records)
    q_imp["q_importance"] = minmax_norm(q_imp["q_imp_raw"].values)
    q_imp_out = q_imp[["question", "q_importance"]].sort_values("question").reset_index(drop=True)

    q_rel_out.to_csv(os.path.join(OUT_DIR, "q_reliability.csv"), index=False)
    q_imp_out.to_csv(os.path.join(OUT_DIR, "q_importance.csv"), index=False)

    print("[OK] Wrote:")
    print(" - data/priors/q_reliability.csv")
    print(" - data/priors/q_importance.csv")
    print(q_imp_out.head(5))
    print(q_rel_out.head(5))

if __name__ == "__main__":
    main()
