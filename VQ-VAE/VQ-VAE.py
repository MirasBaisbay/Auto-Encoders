import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch.nn.functional as F

os.makedirs('VQ-VAE_results', exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 20
LR_RATE = 3e-4
SAVE_INTERVAL = 5

print("Device:", DEVICE)

# VQ-VAE:
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        # num_embeddings - number of embeddings (K)
        # embedding_dim - dimension of the embeddings         
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        # Creating the embedding table
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim) 
        # Uniformly initializing the embedding table, why? because we want to have a good starting point
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings) 
        # beta term of the loss 
        self._commitment_cost = commitment_cost 
        
    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC because we want to quantize the channels
        # BCHW - batch, channel, height, width
        # BHWC - batch, height, width, channel 
        # contigous() is used to make sure that the tensor is stored in a contiguous chunk of memory
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        # Example: [16, 64, 32, 32] -> [16, 32, 32, 64]
        # flattening will gives us [16*32*32, 64] -> [16384, 64] where 16834 vectors of 64 dimensions
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        # lets say flat_input = (N, 64) and self._embedding.weight = (K, 64)
        # torch.sum(flat_input**2, dim=1, keepdim=True) - (N, 1)
        # torch.sum(self._embedding.weight**2, dim=1) - (K,)
        # torch.matmul(flat_input, self._embedding.weight.t()) - (N, K)
        # distances = (N, 1) + (K,) - 2 * (N, K) = (N, K)
        # why? because we want to find the closest embedding for each vector in the input
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        # Encoding
        # torch.argmin(distances, dim=1) - (N,) finding the index of the closest embedding for each vector in the input
        # unsqueeze(1) - (N, 1)
        # torch.zeros(N, K) - creating a tensor of zeros with shape (N, K)
        # scatter_(1, encoding_indices, 1) - filling the tensor with ones at the indices of the closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and flatten
        # torch.matmul(encodings, self._embedding.weight) - (N, K) * (K, 64) = (N, 64)
        # view(input_shape) - reshaping the tensor to the original shape
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        # Second Loss: F.mse_loss(quantized.detach(), inputs) - detach() is the same as stop gradient in paper. Optimizes the codebook 
        # Third Loss: F.mse_loss(quantized, inputs.detach()) - Optimizes the encoder
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach()) 
        loss = q_latent_loss + self._commitment_cost * e_latent_loss       # Total loss
        
        # passing the quantized tensor through the residual connection from the decoder to the encoder
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
    
# Note that this code snippet is from the Aleksa Gordic's implementation of VQ-VAE which can be found at:
# https://youtu.be/VZFVUrYcig0