# 🌀 Flow Matching in Generative Modeling

This is a minimal repo demonstrating **Flow Matching** — a method to train generative models via supervised learning of ODE trajectories.

---

## 🌊 What is Flow Matching?

**Flow Matching** is a recently proposed method ([Lipman et al., 2022](https://arxiv.org/abs/2210.02747)) for training **generative models** by learning a vector field that transforms samples from a simple distribution (e.g., Gaussian noise) into samples from a complex data distribution (e.g., images, shapes, point clouds).

---

### 🚀 Core Idea

We define a **linear interpolation** between a pair of points:

<p align = "center">
    <img src="https://latex.codecogs.com/svg.image?\color{white}x_t=t\cdot&space;x_0&plus;(1-t)\cdot&space;x_1"/>
</p>

- `x₀` is a sample from the **prior distribution** (e.g., Gaussian noise)
- `x₁` is a sample from the **target distribution** (e.g., real data)
- `xₜ` is a point interpolated between the two
- `t ∈ [0, 1]` is a random time step

We want to **learn a time-dependent vector field `v(xₜ, t)`** that matches the **true constant velocity** from `x₀` to `x₁`, defined as:

<p align="center">
    <img src="https://latex.codecogs.com/svg.image?\color{white}\frac{d&space;x_t}{dt}=x_1-x_0"/>
</p>

This defines a trivial ODE where each point moves at constant velocity along a straight line.

---

### 🧠 Training Objective

We train a neural network `v_θ(xₜ, t)` to match this target velocity using an MSE loss:

<p align="center">
    <img src="https://latex.codecogs.com/svg.image?\color{white}\mathcal{L}=\mathbb{E}_{x_0,x_1,t}\left[\left\|v_\theta(x_t,t)-(x_1-x_0)\right\|^2\right]"/>
</p>

Effectively, the model learns to predict the direction and magnitude each point should move in at any timestep `t` along its trajectory from `x₀` to `x₁`.

---

### 🎯 Sampling

To generate new samples:
1. Sample `x₀ ∼ p₀` (e.g., standard normal distribution)
2. Solve the ODE:

<p align="center">
    <img src="https://latex.codecogs.com/svg.image?\color{white}\frac{d&space;x_t}{dt}=v_\theta(x_t,t)"/>
</p>

from `t = 0` to `t = 1`, using an ODE solver (e.g., Euler, Runge-Kutta).

The resulting `x₁` is a sample from the learned data distribution.

---

## 📊 Results

### Final Reconstructed Data
![Final Reconstructed Data](/results/final_scatter.png)

### Initial Noise Distribution
![Initial Noise](/results/scatter_epoch_0001.png)

---

## 🎥 Visualization

![Flow Matching in Action](results/my_awesome%20(2).gif)

This animation shows how points are moved from the prior (noise) distribution to the data distribution by integrating the learned flow field.

---

## 📄 Citation

This repo is based on the ideas from:

> Y. Lipman, G. Batzolis, and S. Achlioptas.  
> **"Flow Matching for Generative Modeling"**, NeurIPS 2022.  
> [[arXiv:2210.02747](https://arxiv.org/abs/2210.02747)]

---

## 🛠️ Acknowledgments

This repo is a fun and lightweight implementation for understanding the mechanics of flow matching. It is not optimized for performance or scale, but focuses on clarity and visual intuition.