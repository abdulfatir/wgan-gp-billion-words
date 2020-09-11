#  Copyright (c) 2020 Abdul Fatir Ansari. All rights reserved.
#  This work is licensed under the terms of the MIT license.
#  For a copy, see <https://opensource.org/licenses/MIT>.
#
import torch.nn as nn
import torch
import numpy as np

from torch.nn.utils import spectral_norm

class TextResBlock(nn.Module):
    def __init__(self, dim=512, ksize=5, sn=False):
        super().__init__()
        conv1 = nn.Conv1d(dim, dim, ksize, padding=ksize//2)
        if sn:
            conv1 = spectral_norm(conv1)
        conv2 = nn.Conv1d(dim, dim, ksize, padding=ksize//2)
        if sn:
            conv2 = spectral_norm(conv2)
        self.main = nn.Sequential(nn.ReLU(), conv1, nn.ReLU(), conv2)
    def forward(self, x):
        return x + 0.3 * self.main(x)

class TextGenerator(nn.Module):
    def __init__(self, cmaplen, zdim=128, seq_len=32, dim=512):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.linear1 = nn.Linear(zdim, seq_len*dim)
        self.res1 = TextResBlock(dim=dim)
        self.res2 = TextResBlock(dim=dim)
        self.res3 = TextResBlock(dim=dim)
        self.res4 = TextResBlock(dim=dim)
        self.res5 = TextResBlock(dim=dim)
        self.conv1 = nn.Conv1d(dim, cmaplen, 1)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, z):
        h = self.linear1(z)
        h = h.view(-1, self.dim, self.seq_len)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.conv1(h)
        h = torch.transpose(h, 1, 2)
        o = self.softmax(h)
        return o

class TextDiscriminator(nn.Module):
    def __init__(self, cmaplen, zdim=128, seq_len=32, dim=512, sn=False):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(cmaplen, dim, 1)
        if sn:
            self.conv1 = spectral_norm(self.conv1)
        self.res1 = TextResBlock(dim=dim, sn=sn)
        self.res2 = TextResBlock(dim=dim, sn=sn)
        self.res3 = TextResBlock(dim=dim, sn=sn)
        self.res4 = TextResBlock(dim=dim, sn=sn)
        self.res5 = TextResBlock(dim=dim, sn=sn)
        self.linear1 = nn.Linear(seq_len*dim, 1)
        if sn:
            self.linear1 = spectral_norm(self.linear1)
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        h = self.conv1(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = h.view(-1, self.dim * self.seq_len)
        o = self.linear1(h)
        return o
