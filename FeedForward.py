import paddle
import paddle.nn as nn


class FeedForward(nn.Layer):
    def __init__(self, d_model: int = 512, d_ff=2048):
        super().__init__()
        self.lin_to_big = nn.Linear(d_model, d_ff)
        self.lin_to_small = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.lin_to_small(paddle.nn.functional.relu(self.lin_to_big(x)))
