import torch
import torch.nn as nn

# Input img -> Hidden dim -> mean, std -> Parametrization trick -> Decoder -> Output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, latent_dim=20):
        super().__init__()
        # encoder
        self.img2hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden2mean = nn.Linear(hidden_dim, latent_dim)    # mu (mean)
        self.hidden2std = nn.Linear(hidden_dim, latent_dim)    # std (standard deviation)
        
        # decoder
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden2img = nn.Linear(hidden_dim, input_dim)
        
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
    def encoder(self, x):
        h = self.relu(self.img2hidden(x))
        mu, sigma = self.hidden2mean(h), self.softplus(self.hidden2std(h))
        return mu, sigma
        
        
    def decoder(self, z):
        h = self.relu(self.latent2hidden(z))
        return self.sigmoid(self.hidden2img(h))
        
        
    def reparameterize(self, mu, sigma):
        epsilon = torch.randn_like(sigma)
        return mu + epsilon * sigma
    
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        reconstructed = self.decoder(z)
        return reconstructed, mu, sigma
    
    
if __name__ == "__main__":
    x = torch.randn(4, 28*28)
    vae = VariationalAutoEncoder(input_dim=784)
    print(
            f'Input shape: {x.shape}\n',
            f'Output shape: {vae(x)[0].shape}\n',
            f'Mean shape: {vae(x)[1].shape}\n',
            f'Std shape: {vae(x)[2].shape}'
        )