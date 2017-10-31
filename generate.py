# -*- coding: utf-8 -*-
#
# generate.py
#

from __future__ import print_function
import argparse
import os

import torch
from torch.autograd import Variable

from model import Generator

from utils import scale_image

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Image Generation')
parser.add_argument(
    '--weights',
    default='100_celeb_hq_network-snapshot-010403.pth',
    type=str,
    metavar='PATH',
    help='path to PyTorch state dict')
parser.add_argument('--dir', 
                    default='./generated_samples',
                    type=str, 
                    help='directory to save image in')
parser.add_argument('--name', 
                    default='image.png',
                    type=str, 
                    help='name of image')
parser.add_argument('--seed',
                    default=190,
                    type=int,
                    help='seeding for specific image generation')


def run(args):
    
    print('Loading Generator')
    model = Generator()
    model.load_state_dict(torch.load(args.weights))
    
    # Generate a random latent vector
    x = torch.randn(1, 512, 1, 1)
    
    x = Variable(x, volatile=True)
    
    print('Executing forward pass')
    images = model(x)
    
    
    images_np = images.data.numpy().transpose(0, 2, 3, 1)
    image_np = scale_image(images_np[0, ...])
    
    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    plt.figure()
    plt.imsave(os.path.join(args.dir, args.name), image_np)
    print('Saved generated image: {}'.format(os.path.join(args.dir, args.name)))


def main():

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if not args.weights:
        print('No PyTorch state dict path privided. Exiting...')
        return

    run(args)


if __name__ == '__main__':
    main()