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
        self.d_k = d_model // head  # 64
        self.W_Q = nn.Linear(d_model, d_model)  # 只要3个投影, 不是24个
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.linear_output = nn.Linear(d_model, d_model)

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
        kv_src= query if encoder_output is None else encoder_output
        B ,Lq,_=query.shape
        Lk = kv_src.shape[1]

        q = self.W_Q(query).reshape([B, Lq, self.head, self.d_k]).transpose([0, 2, 1, 3])
        k = self.W_K(kv_src).reshape([B, Lk, self.head, self.d_k]).transpose([0, 2, 1, 3])
        v = self.W_V(kv_src).reshape([B, Lk, self.head, self.d_k]).transpose([0, 2, 1, 3])

        scale = paddle.matmul(q, k, transpose_y=True) / math.sqrt(self.d_k)  # [B,H,Lq,Lk]

        # 1) 因果掩码(仅解码器自注意力 mask=True): 保留"过去+对角线"(j<=i), 屏蔽"未来"(j>i)。
        #    注意 triu(diagonal=1) 标记的是"未来"(上三角), 必须 ==0 反转才得到要保留的下三角。
        #    masked_fill 的语义: mask为True的位置保留原值, False的位置填-1e9。
        if mask:
            causal_keep = paddle.triu(paddle.ones([Lq, Lk]), diagonal=1) == 0
            scale = masked_fill(scale, causal_keep, -1e9)

        # 2) key padding 掩码: 从 [B,L,L] 块掩码的第一行取出"哪些key是真实词"(1=真实,0=pad),
        #    得到 [B,1,1,Lk], 能广播到任意 [B,H,Lq,Lk]。取"最后一维==Lk"的那个掩码:
        #    自注意力时它等于序列自身, 交叉注意力时是英文src_mask(Lk=英文长度)。
        #    这样修掉了原来 src_mask.unsqueeze(1)=[B,1,Lk,Lk] 在交叉注意力(Lq!=Lk)广播崩溃的问题。
        for m in (src_mask, tgt_mask):
            if m is not None and m.shape[-1] == Lk:
                key_pad = m[:, 0, :].unsqueeze(1).unsqueeze(2)  # [B,1,1,Lk]
                scale = masked_fill(scale, key_pad, -1e9)
                break

        out = paddle.matmul(nn.functional.softmax(scale, axis=-1), v)  # [B,H,Lq,d_k]
        out = out.transpose([0, 2, 1, 3]).reshape([B, Lq, -1])
        return self.linear_output(out)





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
