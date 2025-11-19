#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import math
import numpy as np
from singa import tensor
from singa import autograd
from singa import layer
from singa import model
from singa.tensor import Tensor


class Transformer(model.Model):
    def __init__(self, src_n_token, tgt_n_token, d_model=512, n_head=8, dim_feedforward=2048, n_layers=6):
        """
        Transformer model
        Args:
            src_n_token: the size of source vocab
            tgt_n_token: the size of target vocab
            d_model: the number of expected features in the encoder/decoder inputs (default=512)
            n_head: the number of heads in the multi head attention models (default=8)
            dim_feedforward: the dimension of the feedforward network model (default=2048)
            n_layers: the number of sub-en(de)coder-layers in the en(de)coder (default=6)
        """
        super(Transformer, self).__init__()

        self.opt = None
        self.src_n_token = src_n_token
        self.tgt_n_token = tgt_n_token
        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers

        # encoder / decoder / linear
        self.encoder = TransformerEncoder(src_n_token=src_n_token, d_model=d_model, n_head=n_head,
                                          dim_feedforward=dim_feedforward, n_layers=n_layers)
        self.decoder = TransformerDecoder(tgt_n_token=tgt_n_token, d_model=d_model, n_head=n_head,
                                          dim_feedforward=dim_feedforward, n_layers=n_layers)

        self.linear3d = Linear3D(in_features=d_model, out_features=tgt_n_token, bias=False)

        self.soft_cross_entropy = layer.SoftMaxCrossEntropy()

    def forward(self, enc_inputs, dec_inputs):
        """
        Args:
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]

        """
        # enc_outputs: [batch_size, src_len, d_model],
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len]
        # dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.linear3d(dec_outputs)
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns

    def train_one_batch(self, enc_inputs, dec_inputs, dec_outputs, pad):
        out, _, _, _ = self.forward(enc_inputs, dec_inputs)
        shape = out.shape[-1]
        out = autograd.reshape(out, [-1, shape])

        out_np = tensor.to_numpy(out)
        preds_np = np.argmax(out_np, -1)

        dec_outputs_np = tensor.to_numpy(dec_outputs)
        dec_outputs_np = dec_outputs_np.reshape(-1)

        y_label_mask = dec_outputs_np != pad
        correct = preds_np == dec_outputs_np
        acc = np.sum(y_label_mask * correct) / np.sum(y_label_mask)
        dec_outputs = tensor.from_numpy(dec_outputs_np)

        loss = self.soft_cross_entropy(out, dec_outputs)
        self.opt(loss)
        return out, loss, acc

    def set_optimizer(self, opt):
        self.opt = opt
