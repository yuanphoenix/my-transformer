import copy

import paddle.nn as nn

from DecoderLayer import DecoderLayer


class Decoder(nn.Layer):

    def __init__(self, num_layers: int = 6):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.LayerList([copy.deepcopy(DecoderLayer()) for _ in range(num_layers)])

    def forward(self, x, encoder_output):
        """
        :param x: shape [batch,seq_legth,d_model]
        """
        for layer in self.decoder_layers:
            x = layer(x, encoder_output)
        return x
