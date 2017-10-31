# -*- coding: utf-8 -*-
#
# latent_interpolate.py
#

from __future__ import print_function
import argparse
import os

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from model import Generator
from utils import LatentDataset, save_images


parser = argparse.ArgumentParser(description='Interpolation')
parser.add_argument(
    '--weights',
    default='100_celeb_hq_network-snapshot-010403.pth',
    type=str,
    metavar='PATH',
    help='path to PyTorch state dict')
parser.add_argument(
    '--output',
    type=str,
    default='./interpolation_samples',
    help='Directory for storing interpolated imaged')
parser.add_argument(
    '--batch_size',
    default=1,
    type=int,
    help='batch size')
parser.add_argument(
    '--num_workers',
    default=1,
    type=int,
    help='number of workers for DataLoader')
parser.add_argument(
    '--nb_latents',
    default=20,
    type=int,
    help='number of latent vectors to generate')
parser.add_argument(
    '--filter',
    default=2,
    type=int,
    help='length of gaussian filter smoothing the latent vectors')
parser.add_argument(
    '--seed',
    default=1230,
    type=int,
    help='Random seed')


def run(args):
    global use_cuda
    
    print('Loading Generator')
    model = Generator()
    model.load_state_dict(torch.load(args.weights))
    
    pin_memory = False
    
    # Generate latent vector
    latent_dataset = LatentDataset(args.nb_latents, args.filter)
    latent_loader = DataLoader(latent_dataset,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=False,
                               pin_memory=pin_memory)
    
    print('Processing')
    for i, data in enumerate(latent_loader):
        data = Variable(data, volatile=True)

        output = model(data)
    
        images_np = output.data.numpy()
    
        save_images(images_np, args.output, i*args.batch_size)


def main():

    args = parser.parse_args()

    if not args.weights:
        print('No PyTorch state dict path privided. Exiting...')
        return

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    run(args)


if __name__ == '__main__':
    main()
