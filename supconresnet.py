import torch.nn as nn
import torchvision
import torch.nn.functional as F
class SupConResNet(nn.Module):
    def __init__(self, encoder, projection_dim, n_features):
        super(SupConResNet, self).__init__()

        self.encoder = encoder
        self.n_features = n_features
        
        """
        We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        """

        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        return z_i, z_j
    

def build_supconresnet(encoder, projection_dim, n_features):
    return SupConResNet(encoder, projection_dim, n_features)