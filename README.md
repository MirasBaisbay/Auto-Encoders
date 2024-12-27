# This repository contains my progress in learning Auto-encoders
## I will start exploring them one-by-one in the following order:

#### 1. Vanilla Auto-Encoders (AEs) [completed]
#### 2. Variational Auto-Encoders (VAEs) [completed]
#### 3. Denoising Auto-Encoders (DAEs) [completed]
#### 3.5. PixelCNN (autoregressive model) [completed]
#### 4. Vector Quantized Variational Autoencoders (VQ-VAEs)


### Note: PixelCNN was implemented because VQ-VAE is the 'combination' of VAE + PixelCNN (autoregressive models)

## TL;DR(so far): 

#### AE: Encoder-Decoder structure which is mirrored, cannot reproduce new images, used only to reconstruction purposes.
#### VAE: everything that AE has + Encoder outputs the distribution (mean, std) and then reparametrization trick is used for backprop mu + epsilon * std, it can produce new images, edit images(by changing the latent space[mean and std of distribution]), and reconstruct the input images.
#### DAE: everything that AE has + during training both the clean and noisy images is used, noisy images are fed into the model but losses are calculated using the clean images, are capable of removing the noises from the images.