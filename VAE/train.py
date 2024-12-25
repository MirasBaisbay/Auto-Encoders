import torch
import torchvision.datasets as datasets
from tqdm import tqdm
import torch.nn as nn
from model import VariationalAutoEncoder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
os.makedirs('samples', exist_ok=True)
os.makedirs('reconstructions', exist_ok=True)

# configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 784
HIDDEN_DIM = 200
LATENT_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4

print(f'Device: {DEVICE}')

# dataset loading
dataset = datasets.MNIST(root='data/', 
                         train=True,
                         transform=transforms.ToTensor(), 
                         download=True)

train_loader = DataLoader(dataset=dataset, 
                          batch_size=BATCH_SIZE,
                          shuffle=True)

model = VariationalAutoEncoder(input_dim=INPUT_DIM, 
                               hidden_dim=HIDDEN_DIM,
                               latent_dim=LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
criterion = nn.BCELoss(reduction='sum')

# training loop
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for idx, (x, _) in enumerate(tqdm(train_loader)):
        x = x.view(-1, 28*28).to(DEVICE)
        x_reconstructed, mu, sigma = model(x)
        
        # reconstruction loss
        reconstruction_loss = criterion(x_reconstructed, x)
        
        # KL divergence loss
        kl_divergence = -0.5 * torch.sum(1.0 + torch.log(sigma**2) - mu**2  - sigma**2)
        
        # total loss
        loss = reconstruction_loss + kl_divergence
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if idx % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()/len(x):.4f}')
            
    with torch.no_grad():
        # Sample from latent space
        sample = torch.randn(64, LATENT_DIM).to(DEVICE)
        sample = model.decoder(sample)
        save_image(sample.view(64, 1, 28, 28),
                  f'samples/sample_epoch_{epoch}.png',
                  nrow=8)
        
        # Save reconstruction of test images
        data = next(iter(train_loader))[0].to(DEVICE)
        recon, _, _ = model(data.view(-1, INPUT_DIM))
        comparison = torch.cat([data[:8],
                              recon.view(-1, 1, 28, 28)[:8]])
        save_image(comparison.cpu(),
                  f'reconstructions/reconstruction_epoch_{epoch}.png',
                  nrow=8)

    print(f'====> Epoch: {epoch} Average loss: {total_loss / len(train_loader.dataset):.4f}')

        




