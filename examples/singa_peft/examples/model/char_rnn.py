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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
'''Train a Char-RNN model using plain text files.
The model is created following https://github.com/karpathy/char-rnn
The train file could be any text file,
e.g., http://cs.stanford.edu/people/karpathy/char-rnn/
'''

from __future__ import division
from __future__ import print_function
from builtins import range
import numpy as np
import sys
import argparse
from tqdm import tqdm

from singa import device
from singa import tensor
from singa import autograd
from singa import layer
from singa import model
from singa import opt


class CharRNN(model.Model):

    def __init__(self, vocab_size, hidden_size=32):
        super(CharRNN, self).__init__()
        self.rnn = layer.LSTM(vocab_size, hidden_size)
        self.cat = layer.Cat()
        self.reshape1 = layer.Reshape()
        self.dense = layer.Linear(hidden_size, vocab_size)
        self.reshape2 = layer.Reshape()
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()
        self.optimizer = opt.SGD(0.01)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def reset_states(self, dev):
        self.hx.to_device(dev)
        self.cx.to_device(dev)
        self.hx.set_value(0.0)
        self.cx.set_value(0.0)

    def initialize(self, inputs):
        batchsize = inputs[0].shape[0]
        self.hx = tensor.Tensor((batchsize, self.hidden_size))
        self.cx = tensor.Tensor((batchsize, self.hidden_size))
        self.reset_states(inputs[0].device)

    def forward(self, inputs):
        x, hx, cx = self.rnn(inputs, (self.hx, self.cx))
        self.hx.copy_data(hx)
        self.cx.copy_data(cx)
        x = self.cat(x)
        x = self.reshape1(x, (-1, self.hidden_size))
        return self.dense(x)

    def train_one_batch(self, x, y):
        out = self.forward(x)
        y = self.reshape2(y, (-1, 1))
        loss = self.softmax_cross_entropy(out, y)
        self.optimizer(loss)
        return out, loss

    def get_states(self):
        ret = super().get_states()
        ret[self.hx.name] = self.hx
        ret[self.cx.name] = self.cx
        return ret

    def set_states(self, states):
        self.hx.copy_from(states[self.hx.name])
        self.hx.copy_from(states[self.hx.name])
        super().set_states(states)