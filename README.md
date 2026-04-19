# Unconditional GAN — MNIST Handwritten Digit Generation
 
A from-scratch implementation of an **Unconditional Generative Adversarial Network (GAN)** in PyTorch, trained on the MNIST dataset to generate realistic handwritten digit images from random noise.
 
The trained model checkpoint is available on Hugging Face: [`maharjanabeeral/unconditional-mnist-gan`](https://huggingface.co/maharjanabeeral/unconditional-mnist-gan)
 
---
 
## How it works
 
A GAN consists of two networks trained in opposition:
 
- **Generator** — takes a random noise vector (latent dim: 100) and maps it to a 28×28 grayscale image
- **Discriminator** — receives either a real MNIST image or a generated one and outputs a score indicating how "real" it looks
The two networks compete: the generator tries to fool the discriminator, while the discriminator tries to tell real from fake. Over time, the generator learns to produce images that are indistinguishable from real MNIST digits.
 
---
 
## Architecture
 
### Generator
 
Input: random noise vector of shape `(batch_size, 100)`
 
| Layer | Output size | Activation |
|---|---|---|
| Linear(100 → 256) | 256 | LeakyReLU + Dropout(0.2) |
| Linear(256 → 512) | 512 | LeakyReLU + Dropout(0.2) |
| Linear(512 → 1024) | 1024 | LeakyReLU + Dropout(0.2) |
| Linear(1024 → 784) | 784 | Tanh |
| Reshape | (1, 28, 28) | — |
 
Output: image tensor in the range `[-1, 1]`, reshaped to `(batch_size, 1, 28, 28)`
 
### Discriminator
 
Input: flattened image of shape `(batch_size, 784)`
 
| Layer | Output size | Activation |
|---|---|---|
| Linear(784 → 1024) | 1024 | LeakyReLU + Dropout(0.2) |
| Linear(1024 → 512) | 512 | LeakyReLU + Dropout(0.2) |
| Linear(512 → 256) | 256 | LeakyReLU + Dropout(0.2) |
| Linear(256 → 1) | 1 | — (raw logit) |
 
Output: a single logit — positive = real, negative = fake
 
---
 
## Training details
 
| Hyperparameter | Value |
|---|---|
| Dataset | MNIST (60,000 training images) |
| Epochs | 200 |
| Batch size | 64 |
| Latent dimension | 100 |
| Learning rate | 0.0001 |
| Optimizer | Adam (both networks) |
| Loss function | BCEWithLogitsLoss |
| Label smoothing | 0.05 (real labels → 0.95, fake → 0.05) |
| Device | CUDA if available, else CPU |
 
**Label smoothing** is applied to stabilise training — real labels are set to `0.95` instead of `1.0`, and fake labels to `0.05` instead of `0.0`. This prevents the discriminator from becoming overconfident and collapsing the generator's gradients early in training.
 
### Training loop (per epoch)
 
**Discriminator step:**
1. Generate fake images from noise (`.detach()` to stop generator gradients)
2. Compute real loss: `BCEWithLogitsLoss(D(real), 0.95)`
3. Compute fake loss: `BCEWithLogitsLoss(D(fake), 0.05)`
4. `discriminator_loss = (real_loss + fake_loss) / 2`
5. Backpropagate and update discriminator
**Generator step:**
1. Generate a fresh batch of fake images
2. Compute `generator_loss = BCEWithLogitsLoss(D(fake), 0.95)` — generator wants discriminator to call its outputs real
3. Backpropagate and update generator
Sample images and loss curves are plotted every 40 epochs during training.
 
---
 
## Setup
 
### Requirements
 
```bash
pip install torch torchvision matplotlib tqdm huggingface_hub
```
 
### Run training
 
```bash
jupyter notebook GAN.ipynb
```
 
Training runs for 200 epochs. Generated samples are displayed every 40 epochs so you can visually track progress.
 
---
 
 
### Loading for inference
 
```python
import torch
 
# Define Generator class (same architecture as above)
generator = Generator(latent_dim=100)
generator.load_state_dict(torch.load("generator.pth", map_location="cpu"))
generator.eval()
 
# Generate images
noise = torch.randn(10, 100)
with torch.no_grad():
    generated = generator(noise)      # shape: (10, 1, 28, 28)
    images = (generated + 1) / 2     # rescale [-1, 1] → [0, 1]
```
 
### Loading the full checkpoint
 
```python
checkpoint = torch.load("gan_full_checkpoint.pth", map_location="cpu")
generator.load_state_dict(checkpoint["generator_state_dict"])
discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
# Also available: checkpoint["epoch"], checkpoint["gen_loss"], checkpoint["disc_loss"]
```
 
---
 
## Key design choices
 
**Why fully connected layers instead of convolutions?** This is an intentional baseline implementation. A fully connected GAN on MNIST is the simplest possible GAN and a clean way to understand the adversarial training loop without the added complexity of convolutional architectures (which are covered separately in the DCGAN project).
 
**Why LeakyReLU?** Standard ReLU kills gradients for negative activations. LeakyReLU (default slope 0.01) keeps a small gradient flowing for negative inputs, which is especially important in the discriminator where dead neurons can prevent the generator from receiving useful feedback.
 
**Why Tanh on the generator output?** MNIST images are normalised to `[-1, 1]` (mean 0.5, std 0.5). Tanh naturally outputs in `[-1, 1]`, matching the data distribution and making training more stable than using Sigmoid.
 
**Why BCEWithLogitsLoss?** It combines a Sigmoid with Binary Cross Entropy in a numerically stable single operation, avoiding floating point issues that occur when computing `log(sigmoid(x))` manually.