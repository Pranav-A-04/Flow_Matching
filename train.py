import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.dataloader import create_checkerboard
from model.neural_net import MLP

def train(model, data, optimizer, num_epochs=100, batch_size=64):
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    losses = []
    for epoch in pbar:
        x1 = data[th.randint(data.size(0), (batch_size,))]
        x0 = th.randn_like(x1)  # Random noise
        target = x1-x0  # Target is the difference
        t = th.rand(x1.size(0), device=x1.device)  # Random timesteps in [0, 1)
        xt = t[:, None]*x1 + (1-t[:, None])*x0  # Linear interpolation
        pred = model(xt, t)
        loss = ((pred - target)**2).mean()  # Mean squared error
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({"Loss": np.mean(losses[-10:])})
        losses.append(loss.item())

def test(model, num_epochs):
    model.eval()
    xt = th.randint(1000, 2) # Random test data
    for i, t in enumerate(th.linspace(0, 1, num_epochs), start=1):
        with th.no_grad():
            pred = model(xt, t.expand(xt.size(0)))
        
        xt = xt + (1 / num_epochs) * pred  # Update xt with prediction
        plt.scatter(xt[:, 0].cpu().numpy(), xt[:, 1].cpu().numpy(), s=1, alpha=0.5)
        plt.title(f"Epoch {i}/{num_epochs}")
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.savefig(f"scatter_epoch_{i:04d}.png")  # Save the plot as an image
        plt.clf()  # Clear the figure for the next plot
    
    return xt
        
def main():
    # Example usage
    data = create_checkerboard(resolution=100)
    print(data.shape)
    data = th.Tensor(data)
    print(data.size(0))
    num_epochs = 100000
    batch_size = 64
    model = MLP(embedding_dim=512, input_dim=2, hidden_dims=[512, 512, 512], output_dim=2)
    optimizer = th.optim.Adam(model.parameters(), lr=4e-4)
    train(model, data, optimizer, num_epochs=num_epochs, batch_size=batch_size)
    th.save(model.state_dict(), "model.pth")
    print("Training complete. Model saved as 'model.pth'.")
    
    #sampling and visualizing
    trained_model = MLP()
    trained_model.load_state_dict(th.load("model.pth"))
    trained_model.eval()
    xt = test(trained_model, num_epochs)
    plt.scatter(xt[:, 0].cpu().numpy(), xt[:, 1].cpu().numpy(), s=1, alpha=0.5)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.savefig(f"final_scatter.png")  # Save the plot as an image
    plt.clf()  # Clear the figure for the next plot
    
if __name__ == "__main__":
    main()

