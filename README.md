# ROME: Ask, Answer, and Detect

This repository contains the code and data for the paper:  
**Ask, Answer, and Detect: Role-Playing LLMs for Personality Detection with Question-Conditioned Mixture-of-Experts**

---

## Repository Structure

```text
ROME/
  scripts/
    common.py
    model_moe.py
    train_answer_pretrain.py
    train_detect.py
    run_experiments.py
    check_release.py
    data/
      datasplit.py
      export_embeddings.py
      compute_question_weights.py
    roleplay/
      generate.py
      mbti_questionnaire.py

  data/
    mbti_1.csv
    questionnaire/
      mbti_questions.txt
    roleplay/
      answers_60_gpt4o.csv
    priors/
      q_importance.csv
      q_reliability.csv
    splits/
      train_uids.txt
      test_uids.txt
    embeddings/              # generated locally (not tracked)

  tests/
    test_paper_alignment.py
  checkpoints/               # generated locally (not tracked)
  requirements.txt
  .gitignore
  README.md
```

---

## Included vs. Not Included

### Included in this repository
- **Raw MBTI dataset**: `data/mbti_1.csv`
- **Questionnaire items**: `data/questionnaire/mbti_questions.txt`
- **Train/test splits (UID lists)**: `data/splits/train_uids.txt`, `data/splits/test_uids.txt`
- **Role-played question-level answers (Ask stage output)**: `data/roleplay/answers_60_gpt4o.csv`
- **Question priors**: `data/priors/q_importance.csv`, `data/priors/q_reliability.csv`
- All source code under `scripts/`

### Not included (generated artifacts)
To keep the repository lightweight, the following large artifacts are **not** tracked:
- **Post/question embeddings** (`.npy`), e.g., `data/embeddings/*`
- **Model checkpoints** (`.pth`), e.g., `checkpoints/*`

You can regenerate them by following the instructions below.

---

## Requirements

- Python `3.10.x`
- Install dependencies:
```bash
pip install -r requirements.txt
```

Notes:

* The code supports **Apple Silicon MPS**, **CUDA**, and **CPU**. Device selection is handled in the scripts.

---

## Data Format

### MBTI dataset

The raw dataset is from Kaggle: [https://www.kaggle.com/datasnaek/mbti-type](https://www.kaggle.com/datasnaek/mbti-type)

In this repository we use `data/mbti_1.csv`, which follows the common Kaggle format with:

* `posts`: user posts concatenated with delimiter `|||` (each segment is treated as a post)
* `type`: MBTI type string (e.g., `INTJ`)
* `uids`: a unique integer id per row (added if missing; ordered from top to bottom)

### Role-play answers (Ask output)

`data/roleplay/answers_60_gpt4o.csv` contains question-level soft answers used for supervision in the Answer stage:

* `uids`
* `Q1 ... Q60`: numeric soft answers (Likert-style)

---

## Quickstart: Run ROME End-to-End

### Step 0 (Optional): Create splits

If you want to regenerate train/validation/test splits:

```bash
python scripts/data/datasplit.py
```

This produces:

* `data/splits/train_uids.txt`
* `data/splits/val_uids.txt`
* `data/splits/test_uids.txt`

If these files already exist, you may skip this step. If `val_uids.txt` is absent, the scripts automatically derive the validation set from the training pool to form a user-level 60/20/20 split.

---

### Step 1: Export embeddings (BERT)

This step generates post embeddings and question embeddings using `bert-base-uncased`.

```bash
python scripts/data/export_embeddings.py
```

Expected outputs (not tracked):

* `data/embeddings/train_post_embeddings.npy`
* `data/embeddings/train_post_index_map.npy`
* `data/embeddings/val_post_embeddings.npy`
* `data/embeddings/val_post_index_map.npy`
* `data/embeddings/test_post_embeddings.npy`
* `data/embeddings/test_post_index_map.npy`
* `data/embeddings/question_embeddings.npy`

---

### Step 2 (Optional): Compute question priors

If you want to recompute question priors:

```bash
python scripts/data/compute_question_weights.py
```

Expected outputs:

* `data/priors/q_importance.csv`
* `data/priors/q_reliability.csv`

This export step is optional. Detect computes question priors directly from the training users.

---

### Step 3: Answer-pretrain (question-level regression)

Train the question-conditioned MoE to predict question-level soft answers.

```bash
python scripts/train_answer_pretrain.py
```

Expected output (not tracked):

* `checkpoints/answer_pretrain/best_pretrain.pth`

---

### Step 4: Detect (final personality prediction)

Train the final detection model, which loads the answer-pretrained checkpoint and performs dimension-wise classification with auxiliary regression.

```bash
python scripts/train_detect.py
```

Expected output (not tracked):

* `checkpoints/detect/best_detect.pth`

During training, the script prints:

* validation macro-F1 across the four dimensions
* test macro-F1 and per-dimension macro-F1 after loading the best checkpoint

---

## Baselines

We compare ROME with a broad set of strong baselines previously reported on MBTI personality detection, spanning multiple modeling paradigms:

1. **Traditional feature-based methods**

   * **XGBoost**: concatenates all posts per user into a single document, builds bag-of-words features, and trains a boosted tree classifier for user-level prediction.

2. **Sequence and hierarchical text encoders**

   * **BiLSTM**: encodes user content with bidirectional LSTMs and aggregates representations to form a user vector.
   * **AttRCNN**: hierarchical RCNN-style encoders with attention mechanisms to extract deep semantic features from social text.
   * **AttnSeq**: hierarchical attention over words and messages (posts) to obtain an aggregated user representation.

3. **PLM-based text-only baselines**

   * **BERTconcat**: concatenates all posts into a single long sequence and encodes it with BERT before classification.
   * **BERTmean**: encodes each post with BERT and aggregates post embeddings by mean pooling to form the user representation.

4. **Cross-post interaction and graph-based fusion**

   * **Transformer-MD**: multi-document Transformer architectures designed to reduce order bias and enable cross-post information access.
   * **TrigNet**: builds a psycholinguistic tripartite graph over posts/words/LIWC-style categories and aggregates signals via graph attention.
   * **D-DGCN**: dynamically induces post graphs and applies deep graph convolutions for order-agnostic evidence fusion.

5. **LLM-enhanced baselines**

   * **TAE**: leverages LLM-generated multi-perspective augmentations and distills them into a lightweight encoder for improved representations.
   * **ETM**: uses LLM-based embeddings and label-side multi-view descriptions aligned to users via a contrastive objective.

If available, the following official repositories are relevant:

* D-DGCN: [https://github.com/djz233/D-DGCN](https://github.com/djz233/D-DGCN)
* ETM: [https://github.com/BUPT-SN/ETM](https://github.com/BUPT-SN/ETM)

---

## Reproducibility Notes

* The provided split files (`data/splits/*.txt`) define the user-level partition. All training stages use the same 60/20/20 train/validation/test split.
* The evaluation follows dimension-wise binary classification for **IE/SN/TF/PJ**, and reports the average of the four per-dimension macro-F1 scores.
* Large artifacts (embeddings/checkpoints) are generated locally and are not committed.

---

## Citation

```bibtex
@inproceedings{rome2026,
  title     = {Ask, Answer, and Detect: Role-Playing LLMs for Personality Detection with Question-Conditioned Mixture-of-Experts},
  author    = {Anonymous},
  booktitle = {Anonymous Submission},
  year      = {2026}
}
```
