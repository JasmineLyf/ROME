import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model_moe import MoE
from common import (ROOT, DATA_DIR, QUESTION_COLS, load_splits, load_answers,
                    answer_targets, load_user_embeddings, question_onehot,
                    select_device, seed_everything, get_question2dim)

ANSWER_PRETRAIN_NUM_EXPERTS = 32
ANSWER_PRETRAIN_HIDDEN_DIM = 1024
EPOCHS = 120
BATCH_SIZE = 64
LR = 5e-4
DEVICE = select_device()


class AnswerPretrainDataset(Dataset):
    """Create pairs lazily; users are partitioned before question expansion."""
    def __init__(self, uids, user_embs, targets, q_emb):
        self.uids = list(uids)
        self.posts = np.stack([user_embs[u] for u in uids]).astype(np.float32)
        self.targets = targets.loc[uids, QUESTION_COLS].to_numpy(dtype=np.float32)
        self.q_emb = np.asarray(q_emb, dtype=np.float32)
        self.onehot = question_onehot()

    def __len__(self):
        return len(self.uids) * len(QUESTION_COLS)

    def __getitem__(self, index):
        user, question = divmod(index, len(QUESTION_COLS))
        return np.concatenate([self.posts[user], self.q_emb[question], self.onehot[question]]), self.targets[user, question]


def prepare_data():
    train_uids, val_uids, _ = load_splits()
    answers = load_answers()
    targets = answer_targets(answers, train_uids + val_uids)
    users = load_user_embeddings()
    q_emb = np.load(DATA_DIR / 'embeddings/question_embeddings.npy')
    if q_emb.ndim != 2 or q_emb.shape[0] != 60:
        raise ValueError('Question embeddings must have shape (60, D).')
    train = AnswerPretrainDataset(train_uids, users, targets, q_emb)
    val = AnswerPretrainDataset(val_uids, users, targets, q_emb)
    input_dim = train.posts.shape[1] + q_emb.shape[1] + 4
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(val, batch_size=BATCH_SIZE), input_dim)


def train_answer_pretrain(seed=42, epochs=EPOCHS, checkpoint=None):
    if epochs < 1:
        raise ValueError('epochs must be positive; no stale checkpoint may be evaluated.')
    seed_everything(seed)
    checkpoint = Path(checkpoint or ROOT / 'checkpoints/answer_pretrain/best_pretrain.pth')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_ld, val_ld, input_dim = prepare_data()
    model = MoE(input_dim, ANSWER_PRETRAIN_NUM_EXPERTS, ANSWER_PRETRAIN_HIDDEN_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # The paper specifies robust L1; SmoothL1 uses the standard unit transition.
    criterion = nn.SmoothL1Loss()
    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss; check inputs and optimization.')
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for xb, yb in val_ld:
                val_total += criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
        val_loss = val_total / len(val_ld.dataset)
        if not np.isfinite(val_loss):
            raise FloatingPointError('Nonfinite validation loss; refusing to use a stale checkpoint.')
        print(f'Epoch {epoch:03d} Train SmoothL1={total / len(train_ld.dataset):.4f} Val SmoothL1={val_loss:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint)
    return checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    train_answer_pretrain(args.seed, args.epochs, args.checkpoint)
