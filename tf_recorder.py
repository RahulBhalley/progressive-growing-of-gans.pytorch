# -*- coding: utf-8 -*-

import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
import utils as utils
from torchvision import datasets
from tensorboardX import SummaryWriter
import os, sys
import utils as utils

__author__ = 'Rahul Bhalley'

class tf_recorder:
    
    def __init__(self):
        os.system('mkdir -p repo/tensorboard')
        for i in range(1000):
            self.targ = 'repo/tensorboard/try_{}'.format(i)
            if not os.path.exists(self.targ):
                self.writer = SummaryWriter(self.targ)
                break
                
    def add_scalar(self, index, val, niter):
        self.writer.add_scalar(index, val, niter)

    def add_scalars(self, index, group_dict, niter):
        self.writer.add_scalar(index, group_dict, niter)

    def add_image_grid(self, index, ngrid, x, niter):
        grid = utils.make_image_grid(x, ngrid)
        self.writer.add_image(index, grid, niter)

    def add_image_single(self, index, x, niter):
        self.writer.add_image(index, x, niter)

    def add_graph(self, index, x_input, model):
        torch.onnx.export(model, x_input, os.path.join(self.targ, "{}.proto".format(index)), verbose=True)
        self.writer.add_graph_onnx(os.path.join(self.targ, "{}.proto".format(index)))

    def export_json(self, out_file):
        self.writer.export_scalars_to_json(out_file)

