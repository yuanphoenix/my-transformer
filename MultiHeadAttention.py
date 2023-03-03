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
        self.linear_list = [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(head * 3)]
        self.linear_output = nn.Linear(d_model * head, d_model)

    def forward(self, x, encoder_output: Tensor = None, mask=False,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None):

        """
        完全按照论文的思路来实现
        :param tgt_mask:
        :param src_mask:
        :param x:
        :param encoder_output:来自encoder的输出
        :return:
        """
        attention_list = []
        for index, linear in enumerate(self.linear_list):
            if index % 3 == 0:
                # query永远来自于自家
                query = linear(x)
            elif index % 3 == 1:
                # 对于key来说，编码器没什么好说的；解码器中间的多头注意力，key和value都来自编码器的输出
                z = x if encoder_output is None else encoder_output
                key = linear(z)
            else:
                z = x if encoder_output is None else encoder_output
                value = linear(z)
                attention_list.append(attention(query, key, value, self.head, src_mask, tgt_mask, mask=mask))
        x = paddle.concat(attention_list, axis=-1)
        return self.linear_output(x)


def attention(query: Tensor, key: Tensor, value: Tensor, head: int,
              src_mask=None,
              tgt_mask=None,
              mask=False) -> Tensor:
    """
    计算 Attention 的函数
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
    # paddle的转置操作真奇葩,好像tf也是这样子？
    scale = paddle.matmul(query, paddle.transpose(key, [0, 2, 1]))
    scale = scale / math.sqrt(dk)
    if src_mask is not None and tgt_mask is not None:
        q_sen_length = scale.shape[-2]
        k_sen_length = scale.shape[-1]
        batch_size = scale.shape[0]
        result = []
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
        scale = masked_fill(scale, src_mask, -1e9)
    elif tgt_mask is not None:
        scale = masked_fill(scale, tgt_mask, -1e9)

    if mask:
        # 这里有一个下三角，只有Docoder才会进入，但是我们这里的scale是一个[batch,tgt_length,tgt_length]
        #
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
    从官方抄的代码，哈哈
    :param x:
    :param mask:
    :param value:
    :return:
    """
    mask = paddle.cast(mask, 'bool')
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, x, y)
