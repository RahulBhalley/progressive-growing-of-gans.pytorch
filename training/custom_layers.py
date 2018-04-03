# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.nn.init import kaiming_normal, calculate_gain
from PIL import Image
import numpy as np
import copy

__author__ = 'Rahul Bhalley'

class ConcatTable(nn.Module):
    '''Concatination of two layers into vector
    '''
    def __init__(self, layer1, layer2):
        super(ConcatTable, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x):
        return [self.layer1(x), self.layer2(x)]

class Flatten(nn.Module):
    '''Flattens the convolution layer
    '''
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class FadeInLayer(nn.Module):
    '''The layer fades in to the network with `alpha` value slowing entering in to existence
    '''
    def __init__(self, config):
        super(FadeInLayer, self).__init__()
        self.alpha = 0.0

    def update_alpha(self, delta):
        self.alpha = self.alpha + delta
        self.alpha = max(0, min(self.alpha, 1.0))

    # input `x` to `forward()` is output from `ConcatTable()`
    def forward(self, x):
        # `x[0]` is `prev_block` output faded out of existence with 1.0 - `alpha`
        # `x[1]` is `next_block` output faded in to existence with `alpha`
        # This is becasue `alpha` increases linearly
        # Both `x[0]` and `x[1]` outputs 3-dim tensor (last block is `to_rgb_block`)
        # So `add()` can work effectively and produce one weighted output
        return torch.add(x[0].mul(1.0 - self.alpha), x[1].mul(self.alpha))  # outputs one value

class MinibatchSTDConcatLayer(nn.Module):
    '''
    '''
    def __init__(self, averaging='all'):
        super(MinibatchSTDConcatLayer, self).__init__()
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode'%self.averaging
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def forward(self, x):
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)
        vals = self.adjusted_std(x, dim=0, keepdim=True)
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)
        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = mean(vals, axis=[2,3], keepdim=True)             # torch.mean(torch.mean(vals, 2, keepdim=True), 3, keepdim=True)
        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = mean(x, [0,2,3], keepdim=True)                   # torch.mean(torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:                                                           # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1]/self.n, self.shape[2], self.shape[3])
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)

    def __repr__(self):
        return self.__class__.__name__ + '(averaging = {})'.format(self.averaging)

class PixelwiseNormLayer(nn.Module):
    '''
    '''
    def __init__(self):
        super(PixelwiseNormLayer, self).__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x / (torch.mean(x ** 2, dim=1, keepdim=True) + self.eps) ** 0.5

class EqualizedConv2d(nn.Module):
    '''Equalize the learning rate for convolotional layer
    '''
    def __init__(self, c_in, c_out, k_size, stride, pad, bias=False):
        super(EqualizedConv2d, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)
        kaiming_normal(self.conv.weight, a=calculate_gain('conv2d'))
        # Scaling the weights for equalized learning
        conv_w = self.conv.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = (torch.mean(self.conv.weight.data ** 2)) ** 0.5
        self.conv.weight.data.copy_(self.conv.weight.data / self.scale)     # for equalized learning rate

    def forward(self, x):
        x = self.conv(x.mul(self.scale))
        return x + self.bias.view(1, -1, 1, 1).expand_as(x)

class EqualizedDeconv2d(nn.Module):
    '''Equalize the learning rate for transpose convolotional layer
    '''
    def __init__(self, c_in, c_out, k_size, stride, pad):
        super(EqualizedDeconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False)
        kaiming_normal(self.deconv.weight, a=calculate_gain('conv2d'))
        # Scaling the weights for equalized learning
        deconv_w = self.deconv.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = (torch.mean(self.deconv.weight.data ** 2)) ** 0.5
        self.deconv.weight.data.copy_(self.deconv.weight.data / self.scale)

    def forward(self, x):
        x = self.deconv(x.mul(self.scale))
        return x + self.bias.view(1, -1, 1, 1).expand_as(x)

class EqualizedLinear(nn.Module):
    '''Equalize the learning rate for linear layer
    '''
    def __init__(self, c_in, c_out):
        super(EqualizedLinear, self).__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        kaiming_normal(self.linear.weight, a=calculate_gain('linear'))
        # Scaling the weights for equalized learning
        linear_w = self.linear.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = (torch.mean(self.linear.weight.data ** 2)) ** 0.5
        self.linear.weight.data.copy_(self.linear.weight.data / self.scale)

    def forward(self, x):
        x = self.linear(x.mul(self.scale))
        return x + self.bias.view(1, -1).expand_as(x)

class GeneralizedDropout(nn.Module):
    '''
    '''
    def __init__(self, mode='mul', strength=0.4, axes=(0, 1), normalize=False):
        super(GeneralizedDropout, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['out', 'drop', 'prop'], 'Invalid GeneralizedDropout mode' % mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize
        self.gain = None

    def forward(self, x, deterministic=False):
        if deterministic or not self.strength:
            return x

        rnd_shape = [s if axis in self.axes else 1 for axis, s in enumerate(x.size())]
        if self.mode == 'drop':
            p = 1 - self.strength
            rnd = np.random.binomial(1, p=0, size=rnd_shape) / p
        elif self.mode == 'mul':
            rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
        else:
            coef = self.strength * x.size(1) ** 0.5
            rnd - np.random.normal(size=rnd_shape) * coef + 1

        if self.normalize:
            rnd = rnd / np.linalg.norm(rnd, keepdim=True)
        rnd = Variable(torch.from_nunpy(rnd).type(x.data.type()))
        if x.is_cuda:
            rnd = rnd.cuda()
        return x * rnd

    def __repr__(self):
        param_str = '(mode = {0}, strength = {1}, axes = {2}, normalize = {3})'.format(self.mode, self.strength, self.axes, self.normalize)
        return self.__class__.__name__ + param_str
