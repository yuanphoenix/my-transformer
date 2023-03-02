import math
import paddle
import paddle.nn as nn
from paddle import Tensor


class TransformerEmbedding(nn.Layer):
    def __init__(self, vocab_size, d_model=512):
        super(TransformerEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEncoding()

    def forward(self, x: Tensor):
        """

        :param x: tensor对象，疑问，这是什么时候转成Tensor的呢？原版的Transformer是使用Tensor生成的数字，所以他不用考虑这个问题。
        又因为Tensor是无法输入字符串的，所以只能输入字符串对应的数字。或许这就是词表存在的意义。
        :return:
        """
        return self.embedding(x) + math.sqrt(self.d_model)


class PositionalEncoding(nn.Layer):

    def __init__(self, d_model: int = 512, max_seq_length: int = 1000):
        """
        PE(pos,2i) = sin(pos/100002i/dmodel)
        通过公式可以知道，位置编码与原来的字信息毫无关系，独立门户的一套操作
        对于在一句话中的一个字对应的512个维度中，位于偶数位置的使用sin函数，位于基数位置的使用cos函数
        """
        super(PositionalEncoding, self).__init__()
        self.pe = paddle.tensor.zeros([max_seq_length, d_model])
        position = paddle.tensor.arange(0, max_seq_length).unsqueeze(1)
        two_i = paddle.tensor.arange(0, d_model, 2)

        temp = paddle.exp(-1 * two_i * math.log(10000.0) / d_model)
        aab = position * temp
        # position 对应的是词的长度
        self.pe[:, 0::2] = paddle.sin(aab.cast('float32'))
        self.pe[:, 1::2] = paddle.cos(aab.cast('float32'))
        #     pe[max_seq_length, d_model]
        self.pe = self.pe.unsqueeze(0)
        #     pe[1,max_seq_length, d_model]

    def forward(self, x: Tensor):
        """
        词向量+位置编码
        :param x: x应该是一个[bactch,seq_length,d_model]的数据

        """
        self.pe.stop_gradient = True
        return x + self.pe[:, x.shape[1]]
