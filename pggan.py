# -*- coding: utf-8 -*-

from __future__ import print_function

import os, sys
from math import floor, ceil

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms as transforms

from network import *
from config import config
import dataloader as dl
import tf_recorder as tensorboard
import utils as utils

__author__ = 'Rahul Bhalley'

class PGGAN:

    def __init__(self, config):
        self.config = config
        self.use_cuda = False
        self.use_mps = False
        
        if torch.backends.mps.is_available():
            self.use_mps = True
            self.device = torch.device("mps")
            torch.set_default_device(self.device)
        elif torch.cuda.is_available():
            self.use_cuda = True
            self.device = torch.device("cuda")
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device("cpu")
            torch.set_default_tensor_type('torch.FloatTensor')

        print(f"Training on device: {self.device}")

        self.nz = config.nz
        self.optimizer = config.optimizer
        self.resl = 2       # we start with resolution 2^2 = 4
        self.lr = config.lr
        self.eps_drift = config.eps_drift
        self.smoothing = config.smoothing
        self.max_resl = config.max_resl
        self.trns_tick = config.trns_tick
        self.stab_tick = config.stab_tick
        self.TICK = config.TICK
        self.global_iter = 0
        self.global_tick = 0
        self.kimgs = 0
        self.stack = 0
        self.epoch = 0
        self.fadein = {'gen': None, 'dis': None}
        self.complete = {'gen': 0, 'dis': 0}
        self.phase = 'init'
        self.flag_flush_gen = False
        self.flag_flush_dis = False
        self.flag_add_noise = self.config.flag_add_noise
        self.flag_add_drift = self.config.flag_add_drift
        self.gan_type = config.gan_type
        self.gp_lambda = 10  # Gradient penalty lambda for WGAN-GP

        # Network settings
        self.G = Generator(config)
        print('Generator architecture:\n{}'.format(self.G.model))
        self.D = Discriminator(config)
        print('Discriminator architecture:\n{}'.format(self.D.model))
        self.criterion = nn.MSELoss()

        if self.use_cuda or self.use_mps:
            self.criterion = self.criterion.to(self.device)
            if self.use_cuda:
                torch.cuda.manual_seed(config.random_seed)
            if config.n_gpu == 1:
                self.G = nn.DataParallel(self.G).to(self.device)
                self.D = nn.DataParallel(self.D).to(self.device)
            else:
                gpus = []
                for i in range(config.n_gpu):
                    gpus.append(i)
                self.G = nn.DataParallel(self.G, device_ids=gpus).to(self.device)
                self.D = nn.DataParallel(self.D, device_ids=gpus).to(self.device)

        # Define tensors, ship model to cuda, and get dataloader
        self.renew_everything()

        # Tensorboard
        self.use_tb = config.use_tb
        if self.use_tb:
            self.tb = tensorboard.tf_recorder()

    def resl_scheduler(self):
        '''
        This method will schedule image resolution `self.resl` progressively
        It should be called every iteration to ensure real value is updated properly
        Step 1. `trns_tick` -> transition in generator
        Step 2. `stab_tick` -> stabilize
        Step 3: `trns_tick` -> transition in discriminator
        Step 4: `stab_tick` -> stabilize
        '''
        if floor(self.resl) != 2:
            self.trns_tick = self.config.trns_tick
            self.stab_tick = self.config.stab_tick

        self.batchsize = self.loader.batchsize
        delta = 1.0 / (2 * self.trns_tick + 2 * self.stab_tick)
        d_alpha = 1.0 * self.batchsize / self.trns_tick / self.TICK

        # Update `alpha` if fade-in layer exists in `Generator`
        if self.fadein['gen'] is not None:
            if self.resl % 1.0 < self.trns_tick * delta:
                self.fadein['gen'].update_alpha(d_alpha)
                self.complete['gen'] = self.fadein['gen'].alpha * 100
                self.phase = 'gtrns'
            elif self.resl % 1.0 >= self.trns_tick * delta and self.resl % 1.0 < (self.trns_tick + self.stab_tick) * delta:
                self.phase = 'gstab'
        # Update `alpha` if fade-in layer exists in `Discriminator`
        if self.fadein['dis'] is not None:
            if self.resl % 1.0 >= (self.trns_tick + self.stab_tick) * delta and self.resl % 1.0 < (self.stab_tick + self.trns_tick * 2) * delta:
                self.fadein['dis'].update_alpha(d_alpha)
                self.complete['dis'] = self.fadein['dis'].alpha * 100
                self.phase = 'dtrns'
            elif self.resl % 1.0 >= (self.stab_tick + self.trns_tick * 2) * delta and self.phase != 'final':
                self.phase = 'dstab'

        prev_kimgs = self.kimgs
        self.kimgs = self.kimgs + self.batchsize
        if self.kimgs % self.TICK < prev_kimgs % self.TICK:
            self.global_tick = self.global_tick + 1
            # Increase `resl` linearly every tick and
            # grow the network architecture
            prev_resl = floor(self.resl)
            self.resl = self.resl + delta
            self.resl = max(2, min(10.5, self.resl))    # clamping , range: 4 ~ 1024

            #
            # Flush the networks
            #
            if self.flag_flush_gen and self.resl % 1.0 >= (self.trns_tick + self.stab_tick) * delta and prev_resl != 2:
                if self.fadein['gen'] is not None:
                    self.fadein['gen'].update_alpha(d_alpha)
                    self.complete['gen'] = self.fadein['gen'].alpha * 100
                self.flag_flush_gen = False
                self.G.flush_network()  # flush Generator
                print('Generator flushed:\n{}'.format(self.G.model))
                self.fadein['gen'] = None
                self.complete['gen'] = 0.0
                self.phase = 'dtrns'
            elif self.flag_flush_dis and floor(self.resl) != prev_resl and prev_resl != 2:
                if self.fadein['dis'] is not None:
                    self.fadein['dis'].update_alpha(d_alpha)
                    self.complete['dis'] = self.fadein['dis'].alpha * 100
                self.flag_flush_dis = False
                self.D.flush_network()  # flush Discriminator
                print('Discriminator flushed:\n{}'.format(self.D.model))
                self.fadein['dis'] = None
                self.complete['dis'] = 0.0
                if floor(self.resl) < self.max_resl and self.phase != 'final':
                    self.phase = 'gtrns'

            #
            # Grow the networks
            #
            if floor(self.resl) != prev_resl and floor(self.resl) < self.max_resl + 1:
                self.lr = self.lr * float(self.config.lr_decay)
                self.G.grow_network(floor(self.resl))
                self.D.grow_network(floor(self.resl))
                self.renew_everything()
                self.fadein['gen'] = self.G.model.fadein_block
                self.fadein['dis'] = self.D.model.fadein_block
                self.flag_flush_gen = True
                self.flag_flush_dis = True

            if floor(self.resl) >= self.max_resl and self.resl % 1.0 >= (self.stab_tick + self.trns_tick * 2) * delta:
                self.phase = 'final'
                self.resl = self.max_resl + (self.stab_tick + self.trns_tick * 2) * delta

    def renew_everything(self):
        '''Renew the dataloader
        '''
        self.loader = dl.dataloader(self.config, self.device)
        self.loader.renew(min(floor(self.resl), self.max_resl))

        # Define tensors
        self.z = torch.FloatTensor(self.loader.batchsize, self.nz).to(self.device)
        self.x = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize).to(self.device)
        self.x_tilde = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize).to(self.device)
        self.real_label = torch.FloatTensor(self.loader.batchsize).fill_(1).to(self.device)
        self.fake_label = torch.FloatTensor(self.loader.batchsize).fill_(0).to(self.device)

        # Enable device
        if self.use_cuda:
            torch.cuda.manual_seed(config.random_seed)

        # Wrapping `autograd.Variable`
        self.x = self.x.requires_grad_()
        self.x_tilde = self.x_tilde.requires_grad_()
        self.z = self.z.requires_grad_()
        self.real_label = self.real_label.requires_grad_()
        self.fake_label = self.fake_label.requires_grad_()

        # Ship new model to device
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)

        # Setup the optimizer
        betas = (self.config.beta1, self.config.beta2)
        if self.optimizer == 'adam':
            if self.gan_type == 'wgan' or self.gan_type == 'wgan-gp':
                self.opt_g = Adam(self.G.parameters(), lr=self.lr, betas=(0.0, 0.9))
                self.opt_d = Adam(self.D.parameters(), lr=self.lr, betas=(0.0, 0.9))
            else:
                self.opt_g = Adam(self.G.parameters(), lr=self.lr, betas=betas, weight_decay=0.0)
                self.opt_d = Adam(self.D.parameters(), lr=self.lr, betas=betas, weight_decay=0.0)

    def feed_interpolated_input(self, x):
        if self.phase == 'gtrns' and floor(self.resl) > 2 and floor(self.resl) <= self.max_resl:
            alpha = self.complete['gen'] / 100.0
            transform = transforms.Compose( [
                                                transforms.ToPILImage(),
                                                transforms.Resize(size=int(pow(2, floor(self.resl) - 1)), interpolation=0),
                                                transforms.Resize(size=int(pow(2, floor(self.resl))), interpolation=0),
                                                transforms.ToTensor(),
                                            ] )
            x_low = x.clone().add(1).mul(0.5)
            for i in range(x_low.size(0)):
                x_low[i] = transform(x_low[i]).mul(2).add(-1)
            x = torch.add(x.mul(alpha), x_low.mul(1 - alpha))   # interpolated_x
        if self.use_cuda or self.use_mps:
            return x.to(self.device)
        else:
            return x

    def add_noise(self, x):
        if self.flag_add_noise == False:
            return x

        if hasattr(self, '_d_'):
            self._d_ = self._d_ * 0.9 + torch.mean(self.fx_tilde).item() * 0.1
        else:
            self._d_ = 0.0
        strength = 0.2 * max(0, self._d_ - 0.5) ** 2
        z = np.random.randn(*x.size()).astype(np.float32) * strength
        z = torch.from_numpy(z).to(self.device)
        return x + z

    def train(self):
        # noise for test
        self.z_test = torch.FloatTensor(self.loader.batchsize, self.nz).to(self.device)
        self.z_test = self.z_test.requires_grad_(False)
        self.z_test.normal_(0.0, 1.0)

        for step in range(0, self.max_resl + 1 + 5):
            for iter in tqdm(range(0, (self.trns_tick * 2 + self.stab_tick * 2) * self.TICK, self.loader.batchsize)):
                self.global_iter = self.global_iter + 1
                self.stack = self.stack + self.loader.batchsize
                if self.stack > ceil(len(self.loader.dataset)):
                    self.epoch = self.epoch + 1
                    self.stack = int(self.stack % (ceil(len(self.loader.dataset))))

                # Resolution scheduler
                self.resl_scheduler()

                # Zero the gradients
                self.G.zero_grad()
                self.D.zero_grad()

                # Update discriminator
                self.x.data = self.feed_interpolated_input(self.loader.get_batch())
                if self.flag_add_noise:
                    self.x = self.add_noise(self.x)
                self.z.data.resize_(self.loader.batchsize, self.nz).normal_(0.0, 1.0)
                self.x_tilde = self.G(self.z)

                self.fx = self.D(self.x)
                self.fx_tilde = self.D(self.x_tilde.detach())

                if self.gan_type == 'standard':
                    loss_d = self.criterion(self.fx, self.real_label) + self.criterion(self.fx_tilde, self.fake_label)
                elif self.gan_type == 'wgan' or self.gan_type == 'wgan-gp':
                    loss_d = torch.mean(self.fx_tilde) - torch.mean(self.fx)

                    if self.gan_type == 'wgan-gp':
                        # Compute gradient penalty
                        alpha = torch.rand(self.x.size(0), 1, 1, 1).to(self.device)
                        x_hat = (alpha * self.x.data + (1 - alpha) * self.x_tilde.data).requires_grad_(True)
                        fx_hat = self.D(x_hat)
                        grad = torch.autograd.grad(
                            outputs=fx_hat, inputs=x_hat,
                            grad_outputs=torch.ones(fx_hat.size()).to(self.device),
                            create_graph=True, retain_graph=True, only_inputs=True
                        )[0]
                        grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda
                        loss_d += grad_penalty

                # Compute gradients and apply update to parameters
                loss_d.backward()
                self.opt_d.step()

                # Clip weights for WGAN
                if self.gan_type == 'wgan':
                    for p in self.D.parameters():
                        p.data.clamp_(-0.01, 0.01)

                # Update generator
                self.G.zero_grad()
                fx_tilde = self.D(self.x_tilde)
                
                if self.gan_type == 'standard':
                    loss_g = self.criterion(fx_tilde, self.real_label.detach())
                elif self.gan_type == 'wgan' or self.gan_type == 'wgan-gp':
                    loss_g = -torch.mean(fx_tilde)

                # Compute gradients and apply update to parameters
                loss_g.backward()
                self.opt_g.step()

                # Log information
                log_msg = ' [E:{0}][T:{1}][{2:6}/{3:6}]  errD: {4:.4f} | errG: {5:.4f} | [lr:{11:.5f}][cur:{6:.3f}][resl:{7:4}][{8}][{9:.1f}%][{10:.1f}%]'.format(
                    self.epoch, self.global_tick, self.stack, len(self.loader.dataset), loss_d.item(), loss_g.item(), self.resl, int(pow(2,floor(self.resl))), self.phase, self.complete['gen'], self.complete['dis'], self.lr)
                tqdm.write(log_msg)

                # Save the model
                self.snapshot('./repo/model')

                # Save the image grid
                if self.global_iter % self.config.save_img_every == 0:
                    x_test = self.G(self.z_test)
                    os.system('mkdir -p repo/save/grid')
                    utils.save_image_grid(x_test.data, 'repo/save/grid/{}_{}_G{}_D{}.jpg'.format(int(self.global_iter / self.config.save_img_every), self.phase, self.complete['gen'], self.complete['dis']))
                    os.system('mkdir -p repo/save/resl_{}'.format(int(floor(self.resl))))
                    utils.save_image_single(x_test.data, 'repo/save/resl_{}/{}_{}_G{}_D{}.jpg'.format(int(floor(self.resl)), int(self.global_iter / self.config.save_img_every), self.phase, self.complete['gen'], self.complete['dis']))

                # Tensorboard visualization
                if self.use_tb:
                    x_test = self.G(self.z_test)
                    self.tb.add_scalar('data/loss_g', loss_g.item(), self.global_iter)
                    self.tb.add_scalar('data/loss_d', loss_d.item(), self.global_iter)
                    self.tb.add_scalar('tick/lr', self.lr, self.global_iter)
                    self.tb.add_scalar('tick/cur_resl', int(pow(2,floor(self.resl))), self.global_iter)
                    self.tb.add_image_grid('grid/x_test', 4, utils.adjust_dyn_range(x_test.data.float(), [-1, 1], [0, 1]), self.global_iter)
                    self.tb.add_image_grid('grid/x_tilde', 4, utils.adjust_dyn_range(self.x_tilde.data.float(), [-1, 1], [0, 1]), self.global_iter)
                    self.tb.add_image_grid('grid/x_intp', 4, utils.adjust_dyn_range(self.x.data.float(), [-1, 1], [0, 1]), self.global_iter)

    def get_state(self, target):
        if target == 'gen':
            state = {
                'resl': self.resl,
                'state_dict': self.G.state_dict(),
                'optimizer': self.opt_g.state_dict(),
            }
            return state
        elif target == 'dis':
            state = {
                'resl': self.resl,
                'state_dict': self.D.state_dict(),
                'optimizer': self.opt_d.state_dict(),
            }
            return state

    def snapshot(self, path):
        if not os.path.exists(path):
            os.system('mkdir -p {}'.format(path))
        # Save every 100 tick if the network is in stab phase
        ndis = 'dis_R{}_T{}.pth.tar'.format(int(floor(self.resl)), self.global_tick)
        ngen = 'gen_R{}_T{}.pth.tar'.format(int(floor(self.resl)), self.global_tick)
        if self.global_tick % 50 == 0:
            if self.phase == 'gstab' or self.phase == 'dstab' or self.phase == 'final':
                save_path = os.path.join(path, ndis)
                if not os.path.exists(save_path):
                    torch.save(self.get_state('dis'), save_path)
                    save_path = os.path.join(path, ngen)
                    torch.save(self.get_state('gen'), save_path)
                    print('[snapshot] model saved @ {}'.format(path))

    def evaluate(self):
        pass

    def test_growth(self):
        self.G.grow_network(3)
        self.G.flush_network()
        print(self.G.model)


# Perform the training of PGGAN
print('Configuration')
for k, v in vars(config).items():
    print('{}: {}'.format(k, v))

torch.backends.cudnn.benchmark = True   # boost the speed
pggan = PGGAN(config)
pggan.train()
