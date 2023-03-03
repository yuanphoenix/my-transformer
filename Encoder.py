import copy

import paddle.nn as nn

from EncoderLayer import EncoderLayer


class Encoder(nn.Layer):
    def __init__(self, num_layers: int):
        super(Encoder, self).__init__()
        self.layers = nn.LayerList([copy.deepcopy(EncoderLayer()) for _ in range(num_layers)])

    def forward(self, x,src_mask:None):
        for encoder_layer in self.layers:
            x = encoder_layer(x,src_mask)
        return x
