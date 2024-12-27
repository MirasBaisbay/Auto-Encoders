import torch
from torch import nn, optim, cuda, backends
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
from tqdm import tqdm
backends.cudnn.benchmark = True

os.makedirs('PixelCNN_results', exist_ok=True)

DEVICE = torch.device("cuda" if cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 20
LR_RATE = 3e-4
SAVE_INTERVAL = 5

print("Device:", DEVICE)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Creating a mask for convolutional layer with the same shape as the convolutional weights
        -- initialization all of them with ones (torch.ones_like(self.weight))
        -- register_buffer is makes mask persistent but not a trainable parameter (no backpropagation)
        
        _, _, height, width = self.weight.shape
        we are taking the height and width of the weight tensor and ignoring the in and out channels
        
        -- creating a mask:
        self.mask[:, :, height // 2, width // 2 + (mask_type == 'B'):] = 0
        
        height // 2 - is the center of the height
        width // 2 - is the center of the width
        
        mask_type == 'B' - if the mask type is 'B' then we are setting the mask to zero from the center of the width
        to the end of the width
        
        self.mask[:, :, height // 2 + 1:] = 0 - setting the mask to zero from the center of the height to the end of the height
        
        mask_type == 'A' - if the mask type is 'A' then we are setting the mask to zero from the center of the width + 1 
        
        Type A mask (7×7 kernel):    Type B mask (7×7 kernel):
        
            1 1 1 1 0 0 0               1 1 1 1 0 0 0
            1 1 1 1 0 0 0               1 1 1 1 0 0 0
            1 1 1 1 0 0 0               1 1 1 1 0 0 0
            1 1 1 0 0 0 0               1 1 1 1 0 0 0
            0 0 0 0 0 0 0               0 0 0 0 0 0 0
            0 0 0 0 0 0 0               0 0 0 0 0 0 0
            0 0 0 0 0 0 0               0 0 0 0 0 0 0
            
        Type A mask is used for initial layers (because we dont have any information about the pixel we are predicting)
        Type B mask is used for subsequent layers (because we have information about the pixel we are predicting)
        """
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, height, width = self.weight.shape
        
        # creating a mask:
        self.mask[:, :, height // 2, width // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, height // 2 + 1:] = 0
        
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)   # why super().forward(x) ? because we are inheriting from nn.Conv2d and we are overriding the forward method
    
class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # number of filters
        self.fm = 64
        
        self.layers = nn.ModuleList([
            # Initial layers with mask A
            MaskedConv2d('A', 1, self.fm, kernel_size=(7, 7), stride=1, padding=3),
            nn.BatchNorm2d(self.fm),
            nn.ReLU(),
            
            # Subsequent layers with mask B
            *[nn.Sequential(
                MaskedConv2d('B', self.fm, self.fm, kernel_size=(7, 7), stride=1, padding=3),
                nn.BatchNorm2d(self.fm),
                nn.ReLU()
            )for _ in range(7)],
            
            # last layer
            nn.Conv2d(self.fm, 256, 1)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

def train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc='Epoch {}'.format(epoch))
    
    for images, _ in progress_bar:
        images = images.to(DEVICE)
        # Convert pixel values to integers [0,255]
        target = (images[:, 0] * 255).long()
        
        output = model(images)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return total_loss / len(train_loader)


def generate_samples(model, num_samples=64):
    model.eval()
    samples = torch.zeros(num_samples, 1, 28, 28).to(DEVICE)
    
    with torch.no_grad():
        for i in tqdm(range(28), desc="Generating"):
            for j in range(28):
                output = model(samples)
                # get probabilities for next pixel value
                probs = torch.softmax(output[:, :, i, j], dim=1)
                # sample from the distribution
                samples[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.0
                
    return samples


def save_samples(epoch, model, save_path):
    samples = generate_samples(model, num_samples=64)
    save_image(samples, os.path.join(save_path, f'samples_{epoch}.png'), nrow=8)


def main():
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    model = PixelCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
    
    train_losses = []
    
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
        train_losses.append(avg_loss)
        
        print(f'Epoch {epoch} - Loss: {avg_loss:.4f}')
        
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_samples(epoch, model, 'PixelCNN_results')
    
    # Plot training loss
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    
    # Generate samples
    print("Generating samples...")
    save_samples(NUM_EPOCHS, model, 'PixelCNN_results')
    
    print("Training is coomplete. Results saved in 'PixelCNN_results' directory.")
    

if __name__ == "__main__":
    main()
    
    
"""
Some notes about how does PixelCNN work from this article(https://sergeiturukin.com/2017/02/22/pixelcnn.html):

As illustrated it has first convolution layer with mask type ‘A’ that means center pixel
in mask is zeroed, i.e. we guarantee model won’t get access to pixel it is about to predict.
This is really obvious: if we allow pixel-to-be-predicted to be connected to our model then 
the best way to predict its value in the last layer is to learn to mimic it 
(think making center weight equal to one and all others to zero). 
Zeroing center pixel in first layer mask breaks this convenient information flow and forces the model 
to learn to predict the pixel based on previous inputs.

How does masking works in PixelCNN?

masks - are way to restrict information flow from ‘future’ pixels into one we’re predicting

One way (described in a paper) is to use masked convolutions: all we need is just zero out some
weights in convolution filters, like that. It is easy to see, that information from pixels below 
won’t reach target (center) pixel as well as from pixels on the same line to the right of target.

1 1 1 1 1
1 1 1 1 1 
1 1 0 0 0
0 0 0 0 0
0 0 0 0 0

In paper authors actually allow information from R (red) channel go into G (green),
and from R and G go into B. So we have ordering not only within spatial dimensions but 
also within source channels (think colors) too.
"""