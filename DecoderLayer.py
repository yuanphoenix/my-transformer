import paddle.nn as nn
from paddle import Tensor

from FeedForward import FeedForward
from LayerNorm import LayerNorm
from MultiHeadAttention import MultiHeadAttention


class DecoderLayer(nn.Layer):
    def __init__(self):
        """
        解码器部分，
        一个带掩码的多头注意力+norm+残差
        一个不带掩码的多头注意力+norm+残差
        一个前馈神经网络+norm+残差

        """
        super(DecoderLayer, self).__init__()
        self.mask_multi_head_attention = MultiHeadAttention()
        self.multi_head_attention = MultiHeadAttention()
        self.feed_forward = FeedForward()
        self.norm = LayerNorm()

    def forward(self, x, encoder_output: Tensor, src_mask: None, tgt_mask: None):
        """

        :param x: decoder 的输入，他的初始输入应该只有一个标记，但是shape依然是[batch,seq_length,d_model]
        :param encoder_output:编码器的输出
        """
        y = self.mask_multi_head_attention(x, mask=True, tgt_mask=tgt_mask)
        query = x + self.norm(y)
        z = self.multi_head_attention(query, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        z = query + self.norm(z)
        p = self.feed_forward(z)
        output = self.norm(p) + z
        return output
