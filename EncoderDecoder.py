from typing import Optional

import paddle
import paddle.nn as nn
from paddle import Tensor

from Decoder import Decoder
from Embedding import TransformerEmbedding, PositionalEncoding
from Encoder import Encoder
from utils import PAD_ID


class EncoderDecoder(nn.Layer):
    def __init__(self, vocab_size: int, d_model: int = 512):
        super(EncoderDecoder, self).__init__()
        self.layers_nums = 3
        self.embedding = nn.Sequential(
            TransformerEmbedding(vocab_size),
            PositionalEncoding()
        )
        self.encoder = Encoder(self.layers_nums)
        self.decoder = Decoder(self.layers_nums)
        self.linear = nn.Linear(d_model, vocab_size)
        self.soft_max = nn.Softmax()
        # ignore_index 必须等于 utils 里实际填进 label 的 PAD_ID, 两处一致才能跳过 padding
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=PAD_ID)

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
