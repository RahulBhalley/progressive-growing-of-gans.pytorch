# Progressive Growing of Generative Adversarial Network

This is PyTorch implementation of Progressive Growing GANs. The network is trainable on custom image dataset. 

Place your dataset folder inside `data` folder. The training stats are added to `repo` folder as the training progresses.

### Training Configuration

The network training parameters can be configured with following flags.

#### General settings

- `--train_data_root` - set your data sirectory
- `--random_seed` - random seed to reproduce the experiments
- `--n_gpu` - multiple GPU training

#### Training parameters

- `--lr` - learning rate
- `--lr_decay` - learning rate decay at every resolution transition
- `--eps_drift` - coefficient for the drift loss
- `--smoothing` - smoothing factor for smoothed generator
- `--nc` - number of input channel
- `--nz` - input dimension of noise
- `--ngf` - feature dimension of final layer of generator
- `--ndf` - feature dimension of first layer of discriminator
- `--TICK` - 1 tick = 1000 images = (1000/batch_size) iteration
- `--max_resl` - 10-->1024, 9-->512, 8-->256
- `--trns_tick` - transition tick
- `--stab_tick` - stabilization tick

#### Network structure

- `--flag_wn` - use of equalized-learning rate
- `--flag_bn` - use of batch-normalization (not recommended)
- `--flag_pixelwise` - use of pixelwise normalization for generator
- `--flag_gdrop` - use of generalized dropout layer for discriminator
- `--flag_leaky` - use of leaky relu instead of relu
- `--flag_tanh` - use of tanh at the end of the generator
- `--flag_sigmoid` - use of sigmoid at the end of the discriminator
- `--flag_add_noise` - add noise to the real image(x)
- `--flag_norm_latent` - pixelwise normalization of latent vector (z)
- `--flag_add_drift` - add drift loss

#### Optimizer setting

- `--optimizer` - optimizer type
- `--beta1` - beta1 for adam
- `--beta2` - beta2 for adam

#### Display and save setting

- `--use_tb` - enable tensorboard visualization
- `--save_img_every` - save images every specified iteration
- `--display_tb_every` - display progress every specified iteration

### GPU Note

Make sure your machine has CUDA enabled GPU(s) if you want to train on GPUs. Change the `--n_gpu` flag to positive integral value <= available number of GPUs.

### TODO

- WGAN training methodology

### Related Links

- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)

- [Original Implementation in Lasagne/Theano](https://github.com/tkarras/progressive_growing_of_gans)
