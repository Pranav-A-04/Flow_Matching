import os
from model.neural_net import MLP
import torch as th
import matplotlib.pyplot as plt

def test(model, num_steps):
    model.eval()
    xt = th.randn(1000, 2) # Random test data
    
    # Ensure the directory exists
    os.makedirs("sampling_steps", exist_ok=True)
    
    for i, t in enumerate(th.linspace(0, 1, num_steps), start=1):
        with th.no_grad():
            pred = model(xt, t.expand(xt.size(0)))
        
        xt = xt + (1 / num_steps) * pred  # Update xt with prediction
        plt.scatter(xt[:, 0].cpu().numpy(), xt[:, 1].cpu().numpy(), s=4, alpha=0.5)
        plt.title(f"Epoch {i}/{num_steps}")
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.savefig(f"sampling_steps/scatter_epoch_{i:04d}.png")  # Save the plot as an image
        plt.clf()  # Clear the figure for the next plot
    
    return xt

def eval(num_steps=1000):
    #sampling and visualizing
    trained_model = MLP(embedding_dim=512, input_dim=2, hidden_dims=[512, 512, 512], output_dim=2)
    trained_model.load_state_dict(th.load("model.pth"))
    trained_model.eval()
    xt = test(trained_model, num_steps=num_steps)
    plt.scatter(xt[:, 0].cpu().numpy(), xt[:, 1].cpu().numpy(), s=4, alpha=0.5)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.savefig(f"sampling_steps/final_scatter.png")  # Save the plot as an image
    plt.clf()  # Clear the figure for the next plot
    
if __name__ == "__main__":
    eval(num_steps=1000)
    print("Evaluation complete. Check the saved images for results.")