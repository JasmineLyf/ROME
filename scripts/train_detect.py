import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import json
from common import (ROOT, DATA_DIR, load_splits, load_metadata, load_answers,
                    answer_targets, compute_priors, load_user_embeddings,
                    question_onehot, macro_f1, select_device, seed_everything,
                    get_question2dim)
from tqdm.auto import tqdm
from model_moe import MoE
from collections import Counter

# ---- Hyperparameters ----
ANSWER_PRETRAIN_NUM_EXPERTS = 32
ANSWER_PRETRAIN_HIDDEN_DIM  = 1024
TRUNK_HIDDEN                = 512
HEAD_HIDDEN                 = 256
LAMBDA_Q                    = 1.0
LAMBDA_CLS                  = 0.05

# ---- Training config ----
BATCH_SIZE = 32
# Detect epoch count is not specified in the paper; retain the released default.
EPOCHS     = 50
LR         = 1e-4
DEVICE = select_device()

PRETRAIN_CKPT = ROOT / 'checkpoints/answer_pretrain/best_pretrain.pth'
BEST_DETECT_PATH = ROOT / 'checkpoints/detect/best_detect.pth'
QUEST_EMB = DATA_DIR / 'embeddings/question_embeddings.npy'
QUESTION_COLS = [f'Q{i}' for i in range(1, 61)]
dims = ['IE', 'SN', 'TF', 'PJ']

# ---- Model components ----
class GatedFusion(nn.Module):
    def __init__(self, dim_x, dim_s):
        super().__init__()
        self.f_post = nn.Linear(dim_x, dim_x)
        self.f_sub  = nn.Linear(dim_s, dim_x)
        self.gate   = nn.Linear(dim_x + dim_s, dim_x)

    def forward(self, x, s):
        g = torch.sigmoid(self.gate(torch.cat([x, s], 1)))
        return g * self.f_post(x) + (1.0 - g) * self.f_sub(s)

class DetectModel(nn.Module):
    def __init__(self, q_emb, dims_onehot, q_imp_arr, q_unc_arr, user_embs, idxs_by_dim, pretrained_path=PRETRAIN_CKPT):
        super().__init__()

        self.register_buffer("q_emb", torch.tensor(q_emb, dtype=torch.float32))
        self.register_buffer("dims_onehot", torch.tensor(dims_onehot, dtype=torch.float32))
        self.register_buffer("q_imp", torch.tensor(q_imp_arr, dtype=torch.float32))
        self.register_buffer("q_unc", torch.tensor(q_unc_arr, dtype=torch.float32))

        self.user_embs = user_embs
        self.idxs_by_dim = idxs_by_dim

        self.Dp = next(iter(user_embs.values())).shape[0]
        self.inp_dim = self.Dp + q_emb.shape[1] + 4

        # Answer-pretrain MoE (question-level regression)
        self.answer_pretrain = MoE(
            self.inp_dim,
            num_experts=ANSWER_PRETRAIN_NUM_EXPERTS,
            hidden_dim=ANSWER_PRETRAIN_HIDDEN_DIM,
        )
        if pretrained_path is not None and not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Missing pretrain checkpoint: {pretrained_path}")
        if pretrained_path is not None:
            self.answer_pretrain.load_state_dict(torch.load(pretrained_path, map_location="cpu", weights_only=True))

        # Post trunk
        self.trunk = nn.Sequential(
            nn.Linear(self.Dp, TRUNK_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Dimension-specific fusion and heads
        self.fusions = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        for dim in dims:
            k = len(self.idxs_by_dim[dim])
            fusion = GatedFusion(TRUNK_HIDDEN, k)
            self.fusions[dim] = fusion

            self.heads[dim] = nn.Sequential(
                nn.Linear(TRUNK_HIDDEN, HEAD_HIDDEN),
                nn.ReLU(),
                nn.Dropout(0.4 if dim in ("IE", "PJ") else 0.2),
                nn.Linear(HEAD_HIDDEN, 1),
            )

    def forward(self, p_emb):
        B = p_emb.size(0)

        # Answer-pretrain predictions for all 60 questions
        p_e = p_emb.unsqueeze(1).expand(-1, 60, -1)  # (B, 60, Dp)
        q_e = self.q_emb.unsqueeze(0).expand(B, -1, -1).to(p_emb.device)  # (B, 60, Dq)
        d_e = self.dims_onehot.unsqueeze(0).expand(B, -1, -1).to(p_emb.device)  # (B, 60, 4)

        x_w = torch.cat([p_e, q_e, d_e], dim=-1).view(-1, self.inp_dim)  # (B*60, inp_dim)
        preds = self.answer_pretrain(x_w).squeeze(-1).view(B, 60)        # (B, 60)

        trunk_out = self.trunk(p_emb)  # (B, H)

        logits = []
        for dim in dims:
            idxs = self.idxs_by_dim[dim]

            # Equations (7)-(10): multiply fixed priors, then select this construct.
            w_imp = self.q_imp[idxs].unsqueeze(0).expand(B, -1)
            w_unc = (1.0 - self.q_unc[idxs]).unsqueeze(0).expand(B, -1)
            w_new = w_imp * w_unc

            # Dimension summary (k-dim), without q_emb
            summary = preds[:, idxs] * w_new  # (B, k)

            # Shape check for debugging
            assert summary.size(1) == self.fusions[dim].f_sub.in_features, (
                f"{dim}: summary dim {summary.size(1)} != f_sub.in_features {self.fusions[dim].f_sub.in_features}"
            )

            fused = self.fusions[dim](trunk_out, summary)           # (B, H)
            logits.append(self.heads[dim](fused).squeeze(-1))       # (B,)

        return torch.stack(logits, dim=1), preds

# ---- Dataset ----
class MBTIDataset(Dataset):
    def __init__(self, uids, df_meta, df_tgt, user_embs):
        self.uids = uids
        self.df_meta = df_meta
        self.df_tgt = df_tgt
        self.user_embs = user_embs

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        u = self.uids[idx]
        emb = torch.tensor(self.user_embs[u], dtype=torch.float32)

        mbti = self.df_meta.loc[u, "type"]
        cls = [0 if mbti[i] == d[0] else 1 for i, d in enumerate(dims)]
        reg = (self.df_tgt.loc[u, QUESTION_COLS].values.astype(np.float32)
               if self.df_tgt is not None else np.zeros(60, dtype=np.float32))

        return emb, torch.tensor(cls, dtype=torch.float32), torch.tensor(reg, dtype=torch.float32)

# ---- Evaluation ----
def evaluate(model, loader):
    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for emb, cls_t, _ in loader:
            emb, cls_t = emb.to(DEVICE), cls_t.to(DEVICE)
            logits, _ = model(emb)
            preds_bin = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
            all_p.append(preds_bin)
            all_t.append(cls_t.cpu().numpy())

    y_true = np.vstack(all_t)
    y_pred = np.vstack(all_p)
    return macro_f1(y_true, y_pred)

def inspect_prediction_distribution(model, test_loader):
    print("\n== Prediction Distribution ==")
    dim_counts = {d: Counter() for d in dims}
    model.eval()
    with torch.no_grad():
        for emb, _, _ in test_loader:
            emb = emb.to(DEVICE)
            logits, _ = model(emb)
            preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
            for i, dim in enumerate(dims):
                dim_counts[dim].update(preds[:, i].tolist())

    for dim in dims:
        total = sum(dim_counts[dim].values())
        print(f"{dim}: {dict(dim_counts[dim])}  (Total: {total})")


def joint_loss(logits, cls_targets, predictions, answer_targets):
    return (LAMBDA_Q * nn.functional.smooth_l1_loss(predictions, answer_targets)
            + LAMBDA_CLS * nn.functional.binary_cross_entropy_with_logits(logits, cls_targets))


def train_detect(seed=42, epochs=EPOCHS, pretrained_path=PRETRAIN_CKPT,
                 checkpoint=BEST_DETECT_PATH, metrics_path=None):
    if epochs < 1:
        raise ValueError('epochs must be positive; no stale checkpoint may be evaluated.')
    seed_everything(seed)
    train_uids, val_uids, test_uids = load_splits()
    metadata = load_metadata()
    answers = load_answers()
    targets = answer_targets(answers, train_uids)
    # Recompute from training users so legacy prior files cannot leak validation data.
    importance, reliability = compute_priors(answers, metadata, train_uids)
    users = load_user_embeddings()
    q_emb = np.load(QUEST_EMB)
    if q_emb.ndim != 2 or q_emb.shape[0] != 60:
        raise ValueError('Expected question embeddings with shape (60, D).')
    q2dim = get_question2dim()
    idxs_by_dim = {d: [i for i, q in enumerate(QUESTION_COLS) if q2dim[q] == d] for d in dims}
    train_ld = DataLoader(MBTIDataset(train_uids, metadata, targets, users), batch_size=BATCH_SIZE, shuffle=True)
    # Validation and test consume posts and classification labels only.
    val_ld = DataLoader(MBTIDataset(val_uids, metadata, None, users), batch_size=BATCH_SIZE)
    test_ld = DataLoader(MBTIDataset(test_uids, metadata, None, users), batch_size=BATCH_SIZE)
    model = DetectModel(q_emb, question_onehot(), importance, 1 - reliability,
                        users, idxs_by_dim, pretrained_path).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for emb, cls_t, reg_t in tqdm(train_ld, desc=f'Epoch {epoch}'):
            emb, cls_t, reg_t = emb.to(DEVICE), cls_t.to(DEVICE), reg_t.to(DEVICE)
            logits, predictions = model(emb)
            loss = joint_loss(logits, cls_t, predictions, reg_t)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss; check inputs and optimization.')
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(emb)
        scores, avg = evaluate(model, val_ld)
        print(f'Epoch {epoch:02d} Loss={total / len(train_ld.dataset):.4f} Val Macro-F1={avg:.4f} per-dim={scores}')
        if avg > best_val:
            best_val = avg
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    scores, avg = evaluate(model, test_ld)
    result = {'seed': seed, 'per_dimension_macro_f1': dict(zip(dims, scores)),
              'average_macro_f1': avg, 'best_validation_macro_f1': best_val}
    print(json.dumps(result, indent=2))
    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--pretrain-checkpoint', type=Path, default=PRETRAIN_CKPT)
    parser.add_argument('--checkpoint', type=Path, default=BEST_DETECT_PATH)
    parser.add_argument('--metrics', type=Path)
    args = parser.parse_args()
    train_detect(args.seed, args.epochs, args.pretrain_checkpoint, args.checkpoint, args.metrics)
