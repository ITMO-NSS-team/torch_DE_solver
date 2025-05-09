## 📁 `landscape_visualization/` — Loss Landscape Visualization

This module provides an interface for:

* Constructing 2D (or n-dimensional) projected loss landscapes using an autoencoder.

---

## 🚀 Quick Start

### 1. Train the Autoencoder

```bash
python train_burgers.py
```

Trains the autoencoder on a pre-saved trajectory of solver models.

**Outputs:**

* `losses.csv` — loss values per epoch
* `args.json` — run configuration parameters
* `model.pt` — trained autoencoder
* `*.pdf` — trajectory and error plots

---

### 2. Plot the Loss Landscape

```bash
python plot_burgers.py
```

Uses the trained autoencoder to project and visualize the loss landscape.

**Outputs:**

* `map_loss_loss.pdf` — contour plot of the total loss (can be generated for various loss terms)
* `map_relative_error.pdf` — relative projection error
* `map_grid_density.pdf` — model density in latent space
* `summary_*.csv` — error metrics summary

---


## 🗂️ Input Data Structure

* `.pt` files: saved using `torch.save(model.state_dict())`
* Trajectory: files named `model-*.pt`, representing models saved during NN-based PDE solving

---
