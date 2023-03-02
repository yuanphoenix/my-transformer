import paddle.nn as nn
import paddle


class LayerNorm(nn.Layer):
    def __init__(self, d_model: int = 512, eps=1e-6):
        super(LayerNorm, self).__init__()
        paddle.ParamAttr()
        # self.a_2 = paddle.create_parameter(paddle.ones(d_model),dtype=)
        # self.b_2 = paddle.create_parameter(paddle.zeros(d_model))
        self.a_2 = self.create_parameter(shape=[d_model], dtype='float32',
                                         default_initializer=nn.initializer.Constant(1.0))
        self.b_2 = self.create_parameter(shape=[d_model], dtype='float32',
                                         default_initializer=nn.initializer.Constant(0.0))

        self.eps = eps

    def forward(self, x):
        # 就是在统计每个样本所有维度的值，求均值和方差，所以就是在hidden dim上操作
        # 相当于变成[bsz*max_len, hidden_dim], 然后再转回来, 保持是三维
        mean = x.mean(-1, keepdim=True)  # mean: [bsz, max_len, 1]
        std = x.std(-1, keepdim=True)  # std: [bsz, max_len, 1]
        # 注意这里也在最后一个维度发生了广播
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
