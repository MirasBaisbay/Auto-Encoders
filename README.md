# This repository contains my progress in learning Auto-encoders
## I will start exploring them one-by-one in the following order:

#### 1. Vanilla Auto-Encoders (AEs) [completed]
#### 2. Variational Auto-Encoders (VAEs) [completed]
#### 3. Denoising Auto-Encoders (DAEs) [completed]
#### 3.5. PixelCNN (autoregressive model) [completed]
#### 4. Vector Quantized Variational Autoencoders (VQ-VAEs) [Encoder/Decoder needs to be implemented]


### Note: PixelCNN was implemented because VQ-VAE is the 'combination' of VAE + PixelCNN (autoregressive models)

## TL;DR(so far): 

#### AE: Encoder-Decoder structure which is mirrored, cannot reproduce new images, used only to reconstruction purposes.
#### VAE: everything that AE has + Encoder outputs the distribution (mean, std) and then reparametrization trick is used for backprop mu + epsilon * std, it can produce new images, edit images(by changing the latent space[mean and std of distribution]), and reconstruct the input images.
#### DAE: everything that AE has + during training both the clean and noisy images is used, noisy images are fed into the model but losses are calculated using the clean images, are capable of removing the noises from the images.

## Resources to follow to learn about these topics:

#### Overview of Gen AI field: https://youtu.be/2IK3DFHRFfw

### Lectures offered by Universities:
#### 1. Start with Justin Johnson's lectures https://youtu.be/Q3HU2vEhD5Y?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r and https://youtu.be/igP03FXZqgo?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r they will give you a clear idea of what you are going to learn (University of Michigan)
#### 2. Stefano Ermon's Deep Generative Models lectures: https://www.youtube.com/playlist?list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8 (Stanford University)
#### 3. Pieter Abbeel's Deep Unsupervised Learning lectures: https://www.youtube.com/playlist?list=PLwRJQ4m4UJjPIvv4kgBkvu_uygrV3ut_U (UC Berkeley) 

### Youtube tutorials to watch:
#### 1. Why Does Diffusion Work Better than Auto-Regression? https://youtu.be/zc5NTeJbk-k
#### 2. Boltzmann machines https://youtu.be/_bqa_I5hNAo
#### 3. Autoencoders:  https://youtu.be/3jmcHZq3A5s https://youtu.be/hZ4a4NgM3u0 https://youtu.be/zp8clK9yCro
#### 4. Variational Autoencoders: https://youtu.be/qJeaCHQ1k2w https://youtu.be/9zKuYvjFFS8  https://youtu.be/VELQT1-hILo?si=1rJQvnmv78r4JAnO
#### 5. Denoising Autoencoders: https://youtu.be/0V96wE7lY4w?si=Pxpt4iagkiH85NWS
#### 6. VQ-VAE: https://youtu.be/yQvELPjmyn0?si=9jobXk2i6QnZ_REj https://youtu.be/VZFVUrYcig0?si=Fh7iaGwHLWb81FIn