import copy
import math
from typing import Optional

import paddle
import paddle.nn as nn
from paddle import Tensor


class MultiHeadAttention(nn.Layer):
    def __init__(self, d_model: int = 512, head: int = 8):
        super().__init__()
        self.head = head
        """
        MultiHeadAttention在论文中一共出现在了3个地方。在EncoderLayer中一处，在DecoderLay中两处。
        论文中设置了头的数量为8。其实是分别使用网络为q,k,v进行了8次变换。
        这个网络映射过程就是论文中提到的权重变换。
        哈佛论文提出的方法很巧妙，与论文有些出入，所以我并不能理解。
        于是完全按照论文的思路来实现。
        为q,k,v分别进行8次变换，那就是需要有24个网络。
        """
        self.linear_list = [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(head * 3)]
        # 这是经过多头注意力的拼接后，将他们恢复到512维。
        self.linear_output = nn.Linear(d_model * head, d_model)

    def forward(self, query, encoder_output: Optional[Tensor] = None, mask=False,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None):
        """

        :param query: query
        :param encoder_output: encoder的输出
        :param mask: 是否是论文中的MASK-multiheadAttention
        :param src_mask: 来自encoder编码层的掩码，或者是encoder输出的掩码。具体如何判读就是tgt_mask是不是None
        :param tgt_mask: 来自decoder的掩码
        :return:
        """
        attention_list = []
        # 在论文中，self.linear_list的数量是24。
        for index, linear in enumerate(self.linear_list):
            if index % 3 == 0:
                # query永远来自于自家
                query = linear(query)
            elif index % 3 == 1:
                # 对于key来说，编码器没什么好说的；解码器中间的多头注意力，key和value都来自编码器的输出
                # 在编码器中，都是使用query进行权重变换的。
                z = query if encoder_output is None else encoder_output
                key = linear(z)
            else:
                z = query if encoder_output is None else encoder_output
                value = linear(z)
                attention_list.append(attention(query, key, value, self.head, src_mask, tgt_mask, mask=mask))

        query = paddle.concat(attention_list, axis=-1)
        return self.linear_output(query)


def attention(query: Tensor, key: Tensor, value: Tensor, head: int,
              src_mask=None,
              tgt_mask=None,
              mask=False) -> Tensor:
    """
    计算 Attention 的函数。在函数中，计算出来的scale是矩阵乘法的结果，我们为了“不让解码器看到未来的结果”计算出scale后
    将相关的部位置设置为一个极小的数字，这样经过softmax后就几乎为0了，达成了“不让解码器看到未来的结果”的效果。这个是用一个
    下三角矩阵做到的。
    除此之外，其他的矩阵都是遮掩padding的矩阵，不需要“不让解码器看到未来的结果”
    :param src_mask:
    :param tgt_mask:
    :return:
    :param query: shape [batch,seq_length,d_model]
    :param key:同上
    :param value:同上
    :param mask:是否开启掩码矩阵。我们要防止模型看到未来的信息，那么未来的信息来自哪里，当然是解码器的输入啦。所以掩码矩阵的shape为[seq_length,seq_length]
    :param head:头数
    """
    assert query.shape[-1] % head == 0
    dk = query.shape[-1] // head
    # paddle的转置操作真奇葩,好像tf也是这样子
    scale = paddle.matmul(query, paddle.transpose(key, [0, 2, 1]))
    scale = scale / math.sqrt(dk)
    if src_mask is not None and tgt_mask is not None:
        # 这说明是在 DecoderLayer 的第二个多头注意力中。
        q_sen_length = scale.shape[-2]
        k_sen_length = scale.shape[-1]
        batch_size = scale.shape[0]
        result = []
        # 这个需要根据src_mask和tgt_mask生成掩码矩阵
        # src_mask是一个[batch,input_seq_length,input_seq_length]的矩阵，tgt_mask同理，不够这两个矩阵的长度可能会不一样。
        #比如我爱中国，4个字翻译成英语 i love china 就是3个字。
        for index in range(batch_size):
            s = paddle.count_nonzero(src_mask[index])
            lie = int(math.sqrt(s.item()))
            p = paddle.count_nonzero(tgt_mask[index])
            row = int(math.sqrt(p.item()))
            temp = paddle.zeros([q_sen_length, k_sen_length])
            temp[:row, :lie] = 1
            result.append(temp)
        result_mask = paddle.to_tensor(result)
        scale = masked_fill(scale, result_mask, -1e9)

    elif src_mask is not None:
        # Encoderlayer中的mask，也就是为了遮掩住padding的部分
        scale = masked_fill(scale, src_mask, -1e9)
    elif tgt_mask is not None:
        # decoderlayer中的mask，也就是为了遮掩住padding的部分
        scale = masked_fill(scale, tgt_mask, -1e9)

    if mask:
        # 这里有一个下三角，只有decoderlayerr才会进入，但是我们这里的scale是一个[batch,tgt_length,tgt_length]
        seq_length = query.shape[-2]
        down_metric = (paddle.triu(paddle.ones([seq_length, seq_length]), diagonal=1) == 0)
        scale = masked_fill(scale, down_metric, -1e9)
        if tgt_mask is not None:
            assert tgt_mask.shape == scale.shape
            # tgt_mask也是一个[batch，tgt_length,tgt_length]的矩阵
            scale = masked_fill(scale, tgt_mask, -1e9)

    return paddle.matmul(nn.functional.softmax(scale), value)


def masked_fill(x, mask, value):
    """
    从paddle官方抄的代码，哈哈
    :param x:
    :param mask:
    :param value:
    :return:
    """
    mask = paddle.cast(mask, 'bool')
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, x, y)
