from __future__ import print_function

import os
import sys
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init

class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        pred = self.h2_weights(h1)
        
        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            return pred, mae, mse
        else:
            return pred

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)

        weights_init(self)

    def forward(self, x, y=None, reg=0):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y) + ((1/2) * 0.8 * reg)

            # pred = logits.data.max(1, keepdim=True)[1]
            # acc = pred.eq(y.data.view_as(pred)).cpu().sum() / float(y.size()[0])
            # return logits, loss, acc
            #Note: Vic dev
            pred = logits.data.max(1, keepdim=True)[1]
            debug_tmp = pred.eq(y.data.view_as(pred)).sum().float()
            y_size = float(y.size()[0])
            acc = debug_tmp / y_size
            return logits, loss, acc
        else:
            return logits