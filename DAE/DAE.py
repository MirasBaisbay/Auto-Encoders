import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

# Create directory for results
os.makedirs('DAE_results', exist_ok=True)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR_RATE = 3e-4
NUM_EPOCHS = 20
NOISE_FACTOR = 0.5
SAVE_INTERVAL = 5  

print("Device:", DEVICE)

# Data Loading
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, 
                         batch_size=BATCH_SIZE,
                         shuffle=True)

test_loader = DataLoader(test_data,
                        batch_size=BATCH_SIZE,
                        shuffle=False)

# Visualize initial data sample
dataiter = iter(train_loader)
images, labels = next(dataiter)
img = np.squeeze(images[0].numpy())
plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray')
plt.title("Sample Original Image")
plt.axis('off')
plt.show()

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # N, 1, 28, 28 --> N, 16, 14, 14
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            # N, 16, 14, 14 --> N, 32, 7, 7
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            # N, 32, 7, 7 --> N, 64, 1, 1
            nn.Conv2d(32, 64, kernel_size=(7, 7))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(7, 7)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def save_progress_images(epoch, model, test_loader, noise_factor, device):
    """Save a grid of original, noisy, and reconstructed images"""
    model.eval()
    with torch.no_grad():
        sample_size = 8
        test_images = next(iter(test_loader))[0][:sample_size].to(device)
        test_noisy = test_images + noise_factor * torch.randn(*test_images.shape).to(device)
        test_noisy = torch.clip(test_noisy, 0., 1.)
        test_output = model(test_noisy)
        
        # Create comparison grid
        comparison = torch.cat([test_images, test_noisy, test_output], dim=0)
        save_image(comparison.cpu(),
                  f'DAE_results/reconstruction_epoch_{epoch+1}.png',
                  nrow=sample_size)
    model.train()

# Initialize model, criterion, and optimizer
model = DenoisingAutoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)

# Training loop
model.train()
train_losses = []

for epoch in range(NUM_EPOCHS):
    total_train_loss = 0.0
    for batch_idx, (clean_images, _) in enumerate(train_loader):
        clean_images = clean_images.to(DEVICE)
        
        # Add noise to images
        noisy_images = clean_images + NOISE_FACTOR * torch.randn(*clean_images.shape).to(DEVICE)
        noisy_images = torch.clip(noisy_images, 0., 1.)
        
        # Forward pass
        outputs = model(noisy_images)
        loss = criterion(outputs, clean_images)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    # Calculate average loss for the epoch
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_train_loss:.6f}")
    
    # Save progress images
    if (epoch + 1) % SAVE_INTERVAL == 0:
        save_progress_images(epoch, model, test_loader, NOISE_FACTOR, DEVICE)

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('DAE_results/training_loss.png')
plt.show()

# Final evaluation and visualization
model.eval()
with torch.no_grad():
    # Get batch of test images
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(DEVICE)
    
    # Create noisy images
    noisy_images = test_images + NOISE_FACTOR * torch.randn(*test_images.shape).to(DEVICE)
    noisy_images = torch.clip(noisy_images, 0., 1.)
    
    # Get reconstructed images
    reconstructed = model(noisy_images)
    
    # Plot results
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    for i in range(10):
        # Original images
        axes[0,i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
        axes[0,i].axis('off')
        if i == 0:
            axes[0,i].set_title('Original')
        
        # Noisy images
        axes[1,i].imshow(noisy_images[i].cpu().squeeze(), cmap='gray')
        axes[1,i].axis('off')
        if i == 0:
            axes[1,i].set_title('Noisy')
        
        # Reconstructed images
        axes[2,i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
        axes[2,i].axis('off')
        if i == 0:
            axes[2,i].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig('DAE_results/final_results.png')
    plt.show()

# Save final comparison grid
final_comparison = torch.cat([
    test_images[:8],
    noisy_images[:8],
    reconstructed[:8]
], dim=0)

save_image(final_comparison.cpu(),
          'DAE_results/final_comparison_grid.png',
          nrow=8)

print("Training complete. Results saved in 'DAE_results' directory.")