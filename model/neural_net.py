import torch as th
import torch.nn as nn
import torch.nn.functional as F

class NeuralBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu):
        super(NeuralBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
    
    def forward(self, x):
        x = self.activation(self.linear(x))
        return x
    
class MLP(nn.Module):
    def __init__(self, embedding_dim, input_dim, hidden_dims, output_dim, activation=F.relu):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.time_projection = nn.Linear(embedding_dim, hidden_dims[0])
        layers = nn.Sequential(
            *[NeuralBlock(hidden_dims[i], hidden_dims[i + 1], activation)
              for i in range(len(hidden_dims) - 1)] #unpack list into modules using *
        )
        self.layers = layers
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.embedding_dim = embedding_dim
    
    def get_time_embeddings(self, timesteps, embedding_dim):
        factor=10000**((th.arange(
        start=0, end=embedding_dim//2, device=timesteps.device)/(embedding_dim//2)))
        emb = timesteps[:, None].repeat(1, embedding_dim//2) / factor
        emb = th.cat([th.sin(emb), th.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x, timesteps):
        x = self.input_layer(x)
        time_emb = self.get_time_embeddings(timesteps, self.embedding_dim)
        projected_time_emb = self.time_projection(time_emb)
        x = x + projected_time_emb          # Add time embeddings to input
        x = self.layers(x)
        x = self.output_layer(x)
        return x
