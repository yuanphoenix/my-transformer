from typing import Optional

import paddle
import paddle.nn as nn
from paddle import Tensor

from Decoder import Decoder
from Embedding import TransformerEmbedding, PositionalEncoding
from Encoder import Encoder


class EncoderDecoder(nn.Layer):
    def __init__(self, vocab_size: int, d_model: int = 512):
        super(EncoderDecoder, self).__init__()
        self.layers_nums = 6
        self.embedding = nn.Sequential(
            TransformerEmbedding(vocab_size),
            PositionalEncoding()
        )
        self.encoder = Encoder(self.layers_nums)
        self.decoder = Decoder(self.layers_nums)
        self.linear = nn.Linear(d_model, vocab_size)
        self.soft_max = nn.Softmax()
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, x, label, true_label: Optional[Tensor] = None, src_mask=None, tgt_mask=None):
        input_embedding = self.embedding(x)
        label_embedding = self.embedding(label)
        encoder_output = self.encoder(input_embedding, src_mask)
        decoder_output = self.decoder(label_embedding, encoder_output, src_mask, tgt_mask)
        logits = self.linear(decoder_output)
        res_dict = {}
        if true_label is not None:
            loss = self.loss_fct(logits.reshape((-1, logits.shape[-1])),
                                 true_label.reshape((-1,)))
            res_dict['loss'] = loss
        result = self.soft_max(logits)
        max_index = paddle.argmax(result, axis=-1)
        res_dict['logits'] = result
        res_dict['index'] = max_index
        return res_dict
