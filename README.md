# Progressive Growing of Generative Adversarial Network

This is PyTorch implementation of Progressive Growing GANs. The network is trainable on custom image dataset. 

Place your dataset folder inside `data` folder. The training stats are added to `repo` folder as the training progresses.

### Training Configuration

The network training parameters can be configured with following flags.

#### General settings

- `--train_data_root` - Set your data directory
- `--random_seed` - Random seed to reproduce the experiments
- `--n_gpu` - Number of GPUs for multiple GPU training

#### Training parameters

- `--lr` - Learning rate
- `--lr_decay` - Learning rate decay at every resolution transition
- `--eps_drift` - Coefficient for the drift loss
- `--smoothing` - Smoothing factor for smoothed generator
- `--nc` - Number of input channels
- `--nz` - Input dimension of noise
- `--ngf` - Feature dimension of final layer of generator
- `--ndf` - Feature dimension of first layer of discriminator
- `--TICK` - 1 tick = 1000 images = (1000/batch_size) iteration
- `--max_resl` - Maximum resolution (10-->1024, 9-->512, 8-->256)
- `--trns_tick` - Transition tick
- `--stab_tick` - Stabilization tick
- `--gan_type` - GAN training methodology (choices: 'standard', 'wgan', 'wgan-gp')

#### Network structure

- `--flag_wn` - Use of equalized-learning rate
- `--flag_bn` - Use of batch-normalization (not recommended)
- `--flag_pixelwise` - Use of pixelwise normalization for generator
- `--flag_gdrop` - Use of generalized dropout layer for discriminator
- `--flag_leaky` - Use of leaky ReLU instead of ReLU
- `--flag_tanh` - Use of tanh at the end of the generator
- `--flag_sigmoid` - Use of sigmoid at the end of the discriminator
- `--flag_add_noise` - Add noise to the real image(x)
- `--flag_norm_latent` - Pixelwise normalization of latent vector (z)
- `--flag_add_drift` - Add drift loss

#### Optimizer setting

- `--optimizer` - Optimizer type
- `--beta1` - Beta1 for Adam optimizer
- `--beta2` - Beta2 for Adam optimizer

#### Display and save setting

- `--use_tb` - Enable tensorboard visualization
- `--save_img_every` - Save images every specified iteration
- `--display_tb_every` - Display progress every specified iteration

### GPU Note

Make sure your machine has CUDA enabled GPU(s) if you want to train on GPUs. Change the `--n_gpu` flag to positive integral value <= available number of GPUs.

### GAN Training Methodologies

This implementation supports three GAN training methodologies:

1. Standard GAN (default)
2. Wasserstein GAN (WGAN)
3. Wasserstein GAN with Gradient Penalty (WGAN-GP)

To select a specific training methodology, use the `--gan_type` flag:

```
python main.py --gan_type standard
python main.py --gan_type wgan
python main.py --gan_type wgan-gp
```

### Related Links

- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
- [Original Implementation in Lasagne/Theano](https://github.com/tkarras/progressive_growing_of_gans)