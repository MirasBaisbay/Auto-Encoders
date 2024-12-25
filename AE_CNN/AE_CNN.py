import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.ToTensor()

mnist_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

data_loader = DataLoader(dataset=mnist_data,
                         batch_size=64,
                         shuffle=True)

dataiter = iter(data_loader)
images, labels = next(dataiter)
# to analyze our data first then choose the activation function for decoder
print(torch.min(images), torch.max(images)) 


class AutoEncoder(nn.Module):
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
            nn.Sigmoid()    # because our values is in [0, 1] range
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

model = AutoEncoder()
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

num_epochs = 20
outputs = []

for epoch in range(num_epochs):
    model.train()
    for (img, _) in data_loader:
        img = img.to(device)
        recon = model(img)
        loss = criterion(recon, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    outputs.append((epoch, img, recon))
    
    
    
def visualize_reconstructions(outputs):  
    for k in [0, 4, 9, 14, 19]:
        plt.figure(figsize=(20, 4))
        plt.gray()
        
        imgs = outputs[k][1].detach().cpu().numpy()
        recon = outputs[k][2].detach().cpu().numpy()
        
        for i in range(9):
            plt.subplot(2, 9, i + 1)
            plt.imshow(imgs[i].squeeze())
            plt.axis('off')
            if i == 4:
                plt.title(f'Original - Epoch {k+1}')
        
        for i in range(9):
            plt.subplot(2, 9, 9 + i + 1)
            plt.imshow(recon[i].squeeze())
            plt.axis('off')
            if i == 4:
                plt.title('Reconstructed')
        
        plt.tight_layout()
        plt.show()


visualize_reconstructions(outputs) 
