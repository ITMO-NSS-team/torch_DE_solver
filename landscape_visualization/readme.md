## 📁 `landscape_visualization/` — Loss Landscape Visualization

This module provides an interface for:

* Constructing 2D (or n-dimensional) projected loss landscapes using an autoencoder.

---

## 🚀 Quick Start

### 1. 💾 Generate Trajectories

```bash
python burgers_1d_save_model_trejectories.py
```

This script solves the 1D Burgers' equation multiple times using TEDEouS with Adam optimizer and saves model trajectories.

**Outputs:**

* Trajectory folders:
  * `trajectories/burgers/adam_5_starts/adam_0/`, ..., `adam_4/`
  * each containing saved `.pt` models during training
* Result metrics:
  * `results/burgers_1d_adam_save_model_traj_80.csv` — contains RMSE, loss, and training time for each run

Use this script **before training the autoencoder** to ensure the trajectory data exists.

---

### 2. Train the Autoencoder

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

### 3. Plot the Loss Landscape

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

## ⚡ Save Loss Surface data

If you want to **generate a loss surface structure** without rendering visualizations,  
you can directly use the function:

```python
save_equation_loss_surface(...)
```

from:

```
landscape_visualization/_aux/plot_loss_surface.py
```

### Example usage:

```python
from landscape_visualization._aux.plot_loss_surface import PlotLossSurface

plotter = PlotLossSurface(...)
loss_structure = plotter.save_equation_loss_surface(
    u_exact_test, grid_test, grid, domain, equation, boundaries, model_layers
)
```

**Returns a dictionary like:**  
```python
{
    "loss_total": grid_losses_array,
    ...
}
```

---

## 🧠 Direct Model Usage

In addition to reading models from disk (`.pt` files in trajectory folders),  
the visualizator also supports **passing models directly** as `torch.nn.Module` instances.

This is especially useful when:

- You want to generate the loss landscape from models **in memory**
- You’re using a **custom training loop** or generating models dynamically
- You want to avoid saving to disk for speed or space reasons

### How to use:

When calling the `VisualizationModel.train(...)` method, you can pass a list of models via `solver_models`.

```python
from landscape_visualization._aux.visualization_model import VisualizationModel

model = VisualizationModel(...)

# List of torch.nn.Sequential or other compatible models
models = [trained_model_1, trained_model_2, ..., trained_model_N]

model.train(
    optimizer=...,
    epochs=...,
    every_epoch=...,
    batch_size=...,
    resume=False,
    solver_models=models  # Pass models directly here
)
```

### Notes:

- This bypasses loading models from disk.
- Internally, `model.state_dict()` is extracted from each model and used for training the autoencoder.
- The loss computation, latent encoding, and plotting logic remain the same.

---
