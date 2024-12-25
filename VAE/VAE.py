import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image
from pathlib import Path
from typing import Tuple, Dict
import logging
import yaml
from tqdm import tqdm

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 200, latent_dim: int = 20):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Latent space projections
        self.mean_projection = nn.Linear(hidden_dim, latent_dim)
        self.logvar_projection = nn.Linear(hidden_dim, latent_dim)  # Using logvar instead of std
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mean_projection(h), self.logvar_projection(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAETrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.setup_directories()
        self.setup_data()
        self.setup_model()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_directories(self):
        self.sample_dir = Path('samples_VAE')
        self.recon_dir = Path('reconstructions_VAE')
        self.sample_dir.mkdir(exist_ok=True)
        self.recon_dir.mkdir(exist_ok=True)
        
    def setup_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        dataset = datasets.MNIST(
            root='data/',
            train=True,
            transform=transform,
            download=True
        )
        
        self.train_loader = DataLoader(
            dataset=dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
    def setup_model(self):
        self.model = VariationalAutoEncoder(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            latent_dim=self.config['latent_dim']
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        
    def compute_loss(self, x_recon: torch.Tensor, x: torch.Tensor, 
                    mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Reconstruction loss (binary cross entropy)
        recon_loss = nn.functional.binary_cross_entropy(
            x_recon, x, reduction='sum'
        )
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
    
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        
        for idx, (x, _) in enumerate(tqdm(self.train_loader)):
            x = x.view(-1, self.config['input_dim']).to(self.device)
            
            self.optimizer.zero_grad()
            x_recon, mu, logvar = self.model(x)
            loss = self.compute_loss(x_recon, x, mu, logvar)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if idx % self.config['log_interval'] == 0:
                logging.info(f'Epoch: {epoch}, Batch: {idx}, Loss: {loss.item()/len(x):.4f}')
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        logging.info(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
        return avg_loss
    
    def generate_samples(self, epoch: int):
        self.model.eval()
        with torch.no_grad():
            # Generate samples
            sample = torch.randn(64, self.config['latent_dim']).to(self.device)
            sample = self.model.decode(sample)
            save_image(
                sample.view(64, 1, 28, 28),
                self.sample_dir / f'sample_epoch_{epoch}.png',
                nrow=8
            )
            
            # Generate reconstructions
            data = next(iter(self.train_loader))[0].to(self.device)
            recon, _, _ = self.model(data.view(-1, self.config['input_dim']))
            comparison = torch.cat([
                data[:8],
                recon.view(-1, 1, 28, 28)[:8]
            ])
            save_image(
                comparison.cpu(),
                self.recon_dir / f'reconstruction_epoch_{epoch}.png',
                nrow=8
            )
    
    def train(self):
        for epoch in range(self.config['num_epochs']):
            avg_loss = self.train_epoch(epoch)
            self.generate_samples(epoch)
            
            # Could add early stopping here
            
if __name__ == "__main__":
    config = {
        'input_dim': 784,  # 28*28
        'hidden_dim': 200,
        'latent_dim': 20,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'num_epochs': 10,
        'log_interval': 100
    }
    
    trainer = VAETrainer(config)
    trainer.train()