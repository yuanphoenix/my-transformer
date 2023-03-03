import paddle.nn as nn

from FeedForward import FeedForward
from LayerNorm import LayerNorm
from MultiHeadAttention import MultiHeadAttention


class EncoderLayer(nn.Layer):
    def __init__(self):
        """
        编码器的组成部分，一个多头注意力机制+残差+Norm，一个前馈神经网路+残差+Norm，
        """
        super(EncoderLayer, self).__init__()
        self.multi_head = MultiHeadAttention()
        self.feed_forward = FeedForward()
        self.norm = LayerNorm()

    def forward(self, x, src_mask=None):
        """
        :param x: shape [batch,max_length,d_model]
        :return:
        """

        y = self.multi_head(x, src_mask=src_mask)
        y = x + self.norm(y)
        z = self.feed_forward(y)
        z = y + self.norm(z)
        return z
