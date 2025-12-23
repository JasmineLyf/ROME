import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=128):
        super(Router, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        # x: [batch, input_dim]
        h = F.relu(self.fc1(x))           # [batch, hidden_dim]
        logits = self.fc2(h)              # [batch, num_experts]
        gate = F.softmax(logits, dim=-1)  # [batch, num_experts]
        return gate

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, input_dim]
        h = F.relu(self.fc1(x))           # [batch, hidden_dim]
        out = self.fc2(h)                 # [batch, 1]
        return out.squeeze(-1)            # [batch]

class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=4, hidden_dim=128):
        super(MoE, self).__init__()
        self.router = Router(input_dim, num_experts, hidden_dim)
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim) for _ in range(num_experts)
        ])

    def forward(self, x, return_gates=False):
        # x: [batch, input_dim]
        gate = self.router(x)  # [batch, num_experts]
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # [batch, num_experts]
        out = (gate * expert_outs).sum(dim=1)  # [batch]
        if return_gates:
            return out, gate  # 返回 (输出, gate 权重)
        return out

