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
    splits/                  # generated locally (not tracked)
    roleplay/                # generated locally (not tracked)
    priors/                  # generated locally (not tracked)
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
- All source code under `scripts/`

### Not included (generated artifacts)
The following experiment artifacts are generated locally and are **not** included in this repository:
- **Train/validation/test splits (UID lists)**: `data/splits/*`
- **Role-played question-level answers (Ask stage output)**: `data/roleplay/answers_60_gpt4o.csv`
- **Question priors**: `data/priors/q_importance.csv`, `data/priors/q_reliability.csv`
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

The locally generated `data/roleplay/answers_60_gpt4o.csv` contains question-level answers used for supervision in the Answer stage:

* `uids`
* `Q1 ... Q60`: numeric soft answers (Likert-style)

---

## Quickstart: Run ROME End-to-End

### Step 0: Create splits

Generate a user-level 60/20/20 train/validation/test split:

```bash
python scripts/data/datasplit.py
```

This produces:

* `data/splits/train_uids.txt`
* `data/splits/val_uids.txt`
* `data/splits/test_uids.txt`

Use the same split files throughout Ask generation, Answer pretraining, and Detect training. The default split seed is 42. If the split changes, regenerate the corresponding artifacts before retraining.

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

### Step 2: Generate role-play answers (Ask)

Set `OPENAI_API_KEY` in your environment, then run:

```bash
python scripts/roleplay/generate.py
```

This step uses `gpt-4o-2024-08-06` to produce five trials per training/validation user, with temperatures 0.2, 0.3, 0.4, 0.5, and 0.6. It requires API access and incurs API usage charges. Generated answers are saved to:

* `data/roleplay/answers_60_gpt4o.csv`

The script resumes completed trials automatically. It generates no test-user responses. Validation responses are used only to select the Answer checkpoint; they do not contribute gradient updates. Detect evaluation uses posts and class labels without generated answers.

---

### Step 3 (Optional): Export question priors

After generating the Ask answers, export training-only question priors if needed:

```bash
python scripts/data/compute_question_weights.py
```

Expected outputs:

* `data/priors/q_importance.csv`
* `data/priors/q_reliability.csv`

This export step is optional. Detect computes question priors directly from the training users.

---

### Step 4: Answer-pretrain (question-level regression)

Train the question-conditioned MoE to predict question-level soft answers.

```bash
python scripts/train_answer_pretrain.py
```

Expected output (not tracked):

* `checkpoints/answer_pretrain/best_pretrain.pth`

---

### Step 5: Detect (final personality prediction)

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

The paper compares ROME with 16 baseline methods (Table 1), grouped as follows:

1. **Traditional and basic neural models**

   * **XGBoost**: gradient-boosted trees using text features.
   * **BiLSTM**: bidirectional LSTM-based text encoding.
   * **BERTconcat**: concatenates user posts for BERT encoding and classification.
   * **BERTmean**: averages BERT post embeddings to form a user representation.

2. **Deep and structure-enhanced models**

   * **AttRCNN**: attention-based recurrent convolutional text encoding.
   * **AttnSeq**: hierarchical attention over words and posts.
   * **Transformer-MD**: multi-document Transformer modeling across user posts.
   * **TrigNet**: psycholinguistic tripartite graph modeling.
   * **PQ-Net**: psychological knowledge-guided personality modeling.
   * **D-DGCN**: dynamic graph construction and deep graph convolution over posts.
   * **D-DGCN+ℓ₀**: the ℓ₀-regularized D-DGCN variant.
   * **MvP**: multi-view mixture-of-experts for textual personality detection.

3. **LLM-assisted methods**

   * **TAE**: LLM-generated multi-perspective analyses with contrastive learning.
   * **ETM**: LLM-derived long-text embeddings and label-aware alignment.
   * **LL4G**: LLM semantic embeddings for graph construction and user representation learning.
   * **EmoPerso**: conditioned paraphrasing and contextual feature completion for emotion-aware personality representations.

The paper uses GPT-4o-based variants of LL4G and EmoPerso. The scripts in this repository implement ROME; baseline methods are listed here to describe the experimental comparison.

The following baseline repositories are available:

* D-DGCN: [https://github.com/djz233/D-DGCN](https://github.com/djz233/D-DGCN)
* ETM: [https://github.com/BUPT-SN/ETM](https://github.com/BUPT-SN/ETM)
* EmoPerso: [https://github.com/slz0925/EmoPerso.](https://github.com/slz0925/EmoPerso.)
---

## Reproducibility Notes

* Generate split files with `scripts/data/datasplit.py` before running the remaining stages. All training stages use the same 60/20/20 train/validation/test split.
* The evaluation follows dimension-wise binary classification for **IE/SN/TF/PJ**, and reports the average of the four per-dimension macro-F1 scores.
* Splits, Ask answers, question priors, embeddings, and checkpoints are generated locally and are not committed.
* Ask generation uses training labels as offline hints; test labels and test answers are not used for training or prior estimation.

To repeat Answer and Detect training over five seeds after preprocessing:

```bash
python scripts/run_experiments.py
```

This uses seeds 42–46 on the same fixed data split and reports the mean and standard deviation of Macro-F1 in `checkpoints/five_seed_metrics.json`.

Offline tests use temporary synthetic fixtures and do not require API credentials or generated experiment artifacts:

```bash
python -m unittest discover -s tests -v
```

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
