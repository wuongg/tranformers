import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # Mở rộng chiều
        self.linear2 = nn.Linear(d_ff, d_model)  # Thu hẹp lại

    def forward(self, x):
        return self.linear2(nn.functional.relu(self.linear1(x)))  # FFN(x) = W2 * ReLU(W1 * x)