# Progressive-Growing-of-GANs

This is PyTorch implementation of Progressive Growing GANs from original Theano/Lasagne code. This code only has the inference (generator) network and no training. Please download [100-celeb-hq-1024x1024-ours snapshot](https://drive.google.com/drive/folders/0B4qLcYyJmiz0bWJ5bHdKT0d6UXc) which was originally open sourced by the true authors of Progressive Growing GANs. 

To sample an image from learned probability distribution run `python2 generator.py` which samples an image into `generated_samples` folder having 1024x1024 resolution. 

### Example generated image

![alt text](https://raw.githubusercontent.com/rahulbhalley/Progressive-Growing-of-GANs/master/image.png)

##### latent_interpolation.py

Run `python2 latent_interpolation.py` to generate a set of images from latent space generated randomly which can controlled by `--seed` argument in terminal. These images are saved in `interpolation_samples` folder. 

Follwoing flags can be used:

- --weights - path to pretrained PyTorch state dict
- --output - Directory for storing interpolated images
- --batch_size - batch size for DataLoader
- --num_workers - number of workers for DataLoader
- --nb_latents - number of frames to generate
- --filter - gaussian filter length for interpolating latent space
- --seed - random seed for numpy and PyTorch

#### Example interpolation gif

![alt text](https://raw.githubusercontent.com/rahulbhalley/Progressive-Growing-of-GANs/master/anim.gif)

##### transfer_weights.py

To transfer pretrained weights from Lasagne to PyTorch run `python2 transfer_weights.py`. 

# Related Links
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](http://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of//karras2017gan-paper.pdf)
- [Original Implementation in Lasagne/Theano](https://github.com/tkarras/progressive_growing_of_gans)

This code is adopted from [ptrblck](https://github.com/ptrblck)'s implementation. 
