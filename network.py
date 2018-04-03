# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from custom_layers import *
import copy

__author__ = 'Rahul Bhalley'

def deconv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, wn=False, pixel=False, only=False):
    if wn:
        # Why don't we use `EqualizedDeconv2d() here?`
        # It would make more sense.
        layers.append(EqualizedConv2d(c_in, c_out, k_size, stride, pad))
    else:
        layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.ReLU())
        #if bn:
        #    layers.append(nn.BatchNorm2d(c_out))
        if pixel:
            layers.append(PixelwiseNormLayer())
    return layers

def conv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, wn=False, pixel=False, gdrop=True, only=False):
    if gdrop:       layers.append(GeneralizedDropout(mode='prop', strength=0.0))
    if wn:          layers.append(EqualizedConv2d(c_in, c_out, k_size, stride, pad))
    else:           layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:   layers.append(nn.LeakyReLU(0.2))
        else:       layers.append(nn.ReLU())
        if pixel:   layers.append(PixelwiseNormLayer())
    return layers

def linear(layers, c_in, c_out, sig=True, wn=False):
    layers.append(Flatten())
    if wn:      layers.append(EqualizedLinear(c_in, c_out))
    else:       layers.append(nn.Linear(c_in, c_out))
    if sig:     layers.append(nn.Sigmoid())
    return layers

def deepcopy_module(module, target):
    new_module = nn.Sequential()
    for name, m in module.named_children():
        if name == target:
            new_module.add_module(name, m)                  # make new structure
            new_module[-1].load_state_dict(m.state_dict())  # copy weights. -1 'cuz `add_module` appends at end
    return new_module

def soft_copy_param(target_link, source_link, tau):
    '''Soft copy the parameters of a link to another link
    '''
    target_params = dict(target_link.named_parameters())
    for param_name, param in source_link.named_parameters():
        target_params[param_name].data = target_params[param_name].data.mul(1.0 - tau)
        target_params[param_name].data = target_params[param_name].data.add(param.data.mul(tau))

def get_module_names(model):
    names = []
    for key, val in model.state_dict().iteritems():
        name = key.split('.')[0]
        if not name in names:
            names.append(name)
    return names


class Generator(nn.Module):

    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.flag_pixelwise = config.flag_pixelwise
        self.flag_wn = config.flag_wn
        self.flag_leaky = config.flag_leaky
        self.flag_tanh = config.flag_tanh
        self.flag_norm_latent = config.flag_norm_latent
        self.nc = config.nc
        self.nz = config.nz
        self.ngf = config.ngf
        self.layer_name = None
        self.module_names = []
        self.model = self.get_init_gen()

    def first_block(self):
        layers = []
        ndim = self.ngf
        if self.flag_norm_latent:
            layers.append(PixelwiseNormLayer())
        layers = deconv(layers, self.nz, ndim, 4, 1, 3, self.flag_leaky, self.flag_wn, self.flag_pixelwise)
        layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_wn, self.flag_pixelwise)
        return nn.Sequential(*layers), ndim

    def intermediate_block(self, resl):
        '''Return layers according to `resl` 
        '''
        halving = False
        layer_name = 'intermediate_{}x{}_{}x{}'.format(int(pow(2, resl - 1)), int(pow(2, resl - 1)), int(pow(2, resl)), int(pow(2, resl)))
        ndim = self.ngf
        # Update `ndim` according to maximum grown `Generator` to `resl`
        if resl == 3 or resl == 4 or resl == 5:
            halving = False
            ndim = self.ngf
        elif resl == 6 or resl == 7 or resl == 8 or resl == 9 or resl == 10:
            halving = True
            for i in range(int(resl) - 5):  # becuz till `resl` = 5 the `ndim` remains same 
                ndim = ndim / 2
        # Now append layers of upsampling and 2 transpose convolution layer successively
        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        if halving:
            layers = deconv(layers, ndim * 2, ndim, 3, 1, 1, self.flag_leaky, self.flag_wn, self.flag_pixelwise)
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_wn, self.flag_pixelwise)
        else:
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_wn, self.flag_pixelwise)
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_wn, self.flag_pixelwise)
        return nn.Sequential(*layers), ndim, layer_name

    def to_rgb_block(self, c_in):
        layers = []
        layers = deconv(layers, c_in, self.nc, 1, 1, 0, self.flag_leaky, self.flag_wn, self.flag_pixelwise, only=True)
        if self.flag_tanh:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def get_init_gen(self):
        model = nn.Sequential()
        first_block, ndim = self.first_block()
        model.add_module('first_block', first_block)
        model.add_module('to_rgb_block', self.to_rgb_block(ndim))
        self.module_names = get_module_names(model)
        return model

    def grow_network(self, resl):
        new_model = nn.Sequential()
        names = get_module_names(self.model)
        for name, module in self.model.named_children():
            if not name == 'to_rgb_block':                          # except `to_rgb_block`
                new_model.add_module(name, module)                  # make a new structure and
                new_model[-1].load_state_dict(module.state_dict())  # copy pretrained weights

        if resl >= 3 and resl <= 9:
            print('Growing Generator network [{}x{} to {}x{}]... Just a second :)'.format(int(pow(2, resl - 1)), int(pow(2, resl - 1)), int(pow(2, resl)), int(pow(2, resl))))
            low_resl_to_rgb = deepcopy_module(self.model, 'to_rgb_block')   # copy layer with state and structure - deeply copied!
            prev_block = nn.Sequential()
            prev_block.add_module('low_resl_upsample', nn.Upsample(scale_factor=2, mode='nearest'))
            prev_block.add_module('low_resl_to_rgb', low_resl_to_rgb)

            inter_block, ndim, self.layer_name = self.intermediate_block(resl)
            next_block = nn.Sequential()
            next_block.add_module('high_resl_block', inter_block)
            next_block.add_module('high_resl_to_rgb', self.to_rgb_block(ndim))

            new_model.add_module('concat_block', ConcatTable(prev_block, next_block))
            new_model.add_module('fadein_block', FadeInLayer(self.config))
            self.model = None
            self.model = new_model
            self.module_names = get_module_names(self.model)
            print(self.model)

    def flush_network(self):
        try:
            print('Flushing the FadeInLayer and ConcatTable of Generator... Blink twice ;)')
            high_resl_block = deepcopy_module(self.model.concat_block.layer2, 'high_resl_block')
            high_resl_to_rgb = deepcopy_module(self.model.concat_block.layer2, 'high_resl_to_rgb')

            new_model = nn.Sequential()
            for name, module in self.model.named_children():
                if name != 'concat_block' and name != 'fadein_block':
                    new_model.add_module(name, module)                  # make new structure and
                    new_model[-1].load_state_dict(module.state_dict())  # copy pretrained weights

            # now, add the high resolution block
            new_model.add_module(self.layer_name, high_resl_block)
            new_model.add_module('to_rgb_block', high_resl_to_rgb)
            self.model = new_model
            self.module_names = get_module_names(self.model)
            print(self.model)
        except:
            self.model = model

    def freeze_layers(self):
        # although will never use it
        print('Freeze the pretrained weights...')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x.view(x.size(0), -1, 1, 1))
        return x

'''
model = nn.Sequential()
model.add_module('layer1', nn.Linear(10, 10))
model.add_module('layer2', nn.Linear(10, 10))
model.add_module('layer3', nn.Linear(10, 10))
names = get_module_names(model)
print(names)
model[-1].load_state_dict(model[0].state_dict())
#print(model[-1].state_dict()) 
#print(model[0].state_dict())
print(model[0].state_dict()['weight'] == model[-1].state_dict()['weight'])
print(model[0].state_dict()['bias'] == model[-1].state_dict()['bias'])
'''
'''
from config import config

gen = Generator(config)
print(gen)
'''

class Discriminator(nn.Module):

    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.flag_pixelwise = config.flag_pixelwise
        self.flag_wn = config.flag_wn
        self.flag_leaky = config.flag_leaky
        self.flag_sigmoid = config.flag_sigmoid
        self.nz = config.nz
        self.nc = config.nc
        self.ndf = config.ndf
        self.layer_name = None
        self.module_names = []
        self.model = self.get_init_dis()

    def last_block(self):
        ndim = self.ndf
        layers = []
        layers.append(MinibatchSTDConcatLayer())
        layers = conv(layers, ndim + 1, ndim, 3, 1, 1, self.flag_leaky, self.flag_wn, pixel=False)
        layers = conv(layers, ndim, ndim, 4, 1, 0, self.flag_leaky, self.flag_wn, pixel=False)
        layers = linear(layers, ndim, 1, sig=self.flag_sigmoid, wn=self.flag_wn)
        return nn.Sequential(*layers), ndim

    def intermediate_block(self, resl):
        halving = False
        layer_name = 'intermediate_{}x{}_{}x{}'.format(int(pow(2, resl)), int(pow(2, resl)), int(pow(2, resl - 1)), int(pow(2, resl - 1)))
        ndim = self.ndf
        if resl == 3 or resl == 4 or resl == 5:
            halving = False
            ndim = self.ndf
        elif resl == 6 or resl == 7 or resl == 8 or resl == 9 or resl == 10:
            halving = True
            for i in range(int(resl) - 5):
                ndim = ndim / 2
        layers = []
        if halving:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_wn, pixel=False)
            layers = conv(layers, ndim, ndim * 2, 3, 1, 1, self.flag_leaky, self.flag_wn, pixel=False)
        else:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_wn, pixel=False)
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_wn, pixel=False)
        layers.append(nn.AvgPool2d(kernel_size=2))
        return nn.Sequential(*layers), ndim, layer_name

    def from_rgb_block(self, ndim):
        layers = []
        layers = conv(layers, self.nc, ndim, 1, 1, 0, self.flag_leaky, self.flag_wn, pixel=False)
        return nn.Sequential(*layers)

    def get_init_dis(self):
        model = nn.Sequential()
        last_block, ndim = self.last_block()
        model.add_module('from_rgb_block', self.from_rgb_block(ndim))
        model.add_module('last_block', last_block)
        self.module_names = get_module_names(model)
        return model

    def grow_network(self, resl):
        if resl >= 3 and resl <= 9:     # I don't know but we'ew somehow limiting to 9 `resl`
            print('Growing Discriminator network [{}x{} to {}x{}]... Just a second :)'.format(int(pow(2, resl - 1)), int(pow(2, resl -1)), int(pow(2, resl)), int(pow(2, resl))))
            low_resl_from_rgb = deepcopy_module(self.model, 'from_rgb_block')
            prev_block = nn.Sequential()
            prev_block.add_module('low_resl_downsample', nn.AvgPool2d(kernel_size=2))
            prev_block.add_module('low_resl_from_rgb', low_resl_from_rgb)

            inter_block, ndim, self.layer_name = self.intermediate_block(resl)
            next_block = nn.Sequential()
            next_block.add_module('high_resl_from_rgb', self.from_rgb_block(ndim))
            next_block.add_module('high_resl_block', inter_block)

            new_model = nn.Sequential()
            new_model.add_module('concat_block', ConcatTable(prev_block, next_block))
            new_model.add_module('fadein_block', FadeInLayer(self.config))

            names = get_module_names(self.model)
            for name, module in self.model.named_children():
                if not name == 'from_rgb_block':
                    new_model.add_module(name, module)
                    new_model[-1].load_state_dict(module.state_dict())
            self.model = None
            self.model = new_model
            self.module_names = get_module_names(self.model)
            print(self.model)

    def flush_network(self):
        try:
            print('Flushing the FadeInLayer and ConcatTable of Discriminator... Blink twice ;)')
            # Make deep copy and paste
            high_resl_block = deepcopy_module(self.model.concat_block.layer2, 'high_resl_block')
            high_resl_from_rgb = deepcopy_module(self.model.concat_block.layer2, 'high_resl_from_rgb')

            # Add the high resolution block
            new_model = nn.Sequential()
            new_model.add_module('from_rgb_block', high_resl_from_rgb)
            new_model.add_module(self.layer_name, high_resl_block)

            # Add rest
            for name, module in self.model.named_children():
                if name != 'concat_block' and name != 'fadein_block':
                    new_model.add_module(name, module)
                    new_model[-1].load_state_dict(module.state_dict())

            self.model = new_model
            self.module_names = get_module_names(self.model)
            print(self.model)
        except:
            self.model = self.model

    def freeze_layers(self):
        print('Freeze pretrained weights')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x






