```markdown
# Buckley-Leverett Equation Examples

## Overview

This module provides example implementations for solving the Buckley-Leverett equation using Physics-Informed Neural Networks (PINNs) within the `torch_DE_solver` framework. It includes functionalities for defining the problem domain, boundary conditions, neural network architectures, and training procedures specific to the Buckley-Leverett equation. The module showcases the use of various optimization algorithms, including Adam, Natural Gradient Descent (NGD), and Particle Swarm Optimization (PSO), to train the neural networks. It also provides tools for evaluating the trained models against precomputed exact solutions.

## Purpose

The primary purpose of this module is to demonstrate the application of PINNs to solve a specific type of partial differential equation: the Buckley-Leverett equation, which models two-phase flow in porous media. It serves as a practical example of how to utilize the `torch_DE_solver` framework for problem formulation, model training, and result analysis in the context of fluid dynamics. By providing implementations with different optimization techniques, the module aims to facilitate the comparison and development of neural network-based solvers for similar problems.
```