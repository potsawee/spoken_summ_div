import torch
import torch.nn as nn
import numpy as np
import pdb

class AdaptiveGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, batch_first=True, dropout=0.1, device=None):
        super(AdaptiveGRU, self).__init__()
        self.device      = device
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.batch_first = batch_first
        self.dropout     = dropout

        # parameters
        for n in range(num_layers):
            setattr(self, 'grucell_{}'.format(n), nn.GRUCell(input_size, hidden_size, bias) )

        self.dropout_layer = nn.Dropout(p=dropout)
        self.binary_gate = BinaryGate(input_size, hidden_size, hidden_size)

    def forward(self, input, h0, input_lengths):
        batch_size = input.size(0)
        num_utterances = input.size(1)

        output = torch.zeros((batch_size, num_utterances, self.hidden_size)).to(self.device)
        gate_z = torch.zeros((batch_size, num_utterances)).to(self.device)
        segment_indices = [[] for _ in range(batch_size)]

        for bn in range(batch_size):
            ht = [h0 for _ in range(self.num_layers)]
            for t in range(input_lengths[bn]):
                # GRU cell
                xt = input[bn:bn+1, t]

                for n in range(self.num_layers):
                    grucell_n = getattr(self, 'grucell_{}'.format(n))
                    ht[n] = grucell_n(xt, ht[n])

                    # apply dropout --- apart from the output of the last layer
                    if n < self.num_layers - 1:
                        xt = self.dropout_layer(ht[n])
                    else:
                        xt = ht[n]

                ht_n = xt
                output[bn, t] = ht_n[0]

                # binary gate
                if t < input_lengths[bn]-1:
                    xt1 = input[bn:bn+1, t+1]
                    gt = self.binary_gate(xt1, ht_n)
                else:
                    gt = 1.0

                gate_z[bn, t] = gt

                if gt > 0.5: # segmetation & reset GRU cell
                    segment_indices[bn].append(t)
                    ht = [h0 for _ in range(self.num_layers)]
                else: # no segmentation & pass GRU state
                    pass

        return output, segment_indices, gate_z

class BinaryGate(nn.Module):
    def __init__(self, dim1, dim2, hidden_size):
        super(BinaryGate, self).__init__()
        self.linear1 = nn.Linear(dim1, hidden_size, bias=True)
        self.linear2 = nn.Linear(dim2, hidden_size, bias=True)
        self.linearF = nn.Linear(hidden_size, 1,    bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        """computes sigmoid( w^T(W1x1 + W2x2 + b) )
            x1.shape[-1] = dim1
            x2.shape[-1] = dim2
        """
        y = self.linear1(x1) + self.linear2(x2)
        z = self.sigmoid(self.linearF(y))

        return z

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.0, dim=-1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim
        self.reduction = reduction

    def forward(self, pred, target):
        # pred --- logsoftmax already applied
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        if self.reduction == 'mean':
            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        elif self.reduction == 'none':
            return torch.sum(-true_dist * pred, dim=self.dim)
        else:
            raise RuntimeError("reduction mode not supported")
