# torch_DE_solver

---TEDEouS - Torch Exhaustive Differential Equation Solver

[![PyPi](https://badge.fury.io/py/TEDEouS.svg)](https://badge.fury.io/py/TEDEouS)
[![Downloads](https://static.pepy.tech/badge/TEDEouS)](https://pepy.tech/project/TEDEouS)
![License](https://img.shields.io/github/license/ITMO-NSS-team/torch_DE_solver?style=flat&logo=opensourceinitiative&logoColor=white&color=blue)
[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![numpy](https://img.shields.io/badge/NumPy-013243.svg?style={0}&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style={0}&logo=pandas&logoColor=white)
![scipy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style={0}&logo=SciPy&logoColor=white)

---

## Overview

TEDEouS offers an automated approach to solving ordinary and partial differential equations using neural networks, with a focus on making equation discovery more transparent and illustrative. It frames differential equation solving as an optimization problem within a Sobolev space, approximating solutions with parameterized functions and leveraging machine learning for optimization. The core objective is to provide a flexible and modular solver that balances precision and optimization time, moving away from traditional problem-specific solvers. TEDEouS contributes to the methodology described in the associated article by providing tools for automated differential equation solving through parametric approximation optimization, demonstrated through various examples. The solver supports different computational modes and features like mixed precision training, enhancing its applicability in diverse research and application contexts.

---
The purpose of the project:

1.  Make equation discovery more transparent and illustrative
2.  Combine power of pytorch, numerical methods and math overall to conquer and solve ALL XDEs(X={O,P}). There are some examples to provide a little insight to an operator form

---

## Table of Contents

- [Overview](#overview)
- [Content](#content)
- [Algorithms](#algorithms)
- [Core features](#core-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [License](#license)
- [Citation](#citation)

---
## Content

The torch_DE_solver project, or TEDEouS, provides an automated approach to solving differential equations using neural networks. Its core objective is to offer a transparent and illustrative method for equation discovery and solving both ordinary and partial differential equations. The system formulates solutions as optimization problems, approximating them with parameterized functions. Key components include modules for defining problem domains, equations, boundary conditions, neural network models, optimizers, and loss functions. These components work together to train a model that minimizes the error in satisfying the differential equation and boundary conditions. The solver supports various computational modes and features, such as mixed precision training and caching, to enhance performance and flexibility.

---

## Algorithms

The project implements a neural network-based solver for differential equations, framing the solution as an optimization problem. It approximates solutions using neural networks, employing numerical differentiation to evaluate equation residuals. A core aspect involves minimizing a loss function that combines the equation's residual and boundary condition errors, weighted by regularization parameters. The solver supports different computational modes, including finite difference schemes and weak formulations. Causal loss accounts for temporal dependencies. The architecture allows for flexible component interchange, balancing precision and computational efficiency in solving ordinary and partial differential equations.

---

## Core features

*   Solve ODE initial- or boundary-value problems
*   Solve PDE initial-boundary value problems
*   Use variable models and their differentiation methods
*   Faster solution using cache

## Installation

Install torch_DE_solver using one of the following methods:

**Using PyPi:**

```sh
pip install TEDEouS
```

TEDEouS can also be installed with ``pip`` after cloning the repository:

```sh
git clone https://github.com/ITMO-NSS-team/torch_DE_solver.git
cd torch_DE_solver
pip install -r requirements.txt
```

---

## Getting Started

Schroedinger equation example step-by-step https://torch-de-solver.readthedocs.io/en/docs/tedeous/examples/schrodinger.html 

Also, you can refer to various examples in the examples folder. For instance, to run the Legendre polynomial equation example, navigate to the examples folder and execute:

```bash
$ cd examples
$ python example_ODE_Legendre.py
```

---

## Examples

After the TEDEouS is installed the user may refer to various examples that are in examples forlder.

Every example is designed such that the boxplots of the launches are commented and the preliminary results are not shown, but stored in separate folders.

*   Legendre polynomial equation

    ```sh
    python example_ODE_Legendre.py
    ```

    or

    ```sh
    python example_ODE_Legendre_autograd.py
    ```
*   Panleve transcendents (others are placed in 'examples\\to_renew' folder due to the architecture change)

    ```sh
    python example_Painleve_I.py
    ```
*   Wave equation (non-physical conditions for equation discovery problem)

    ```sh
    python example_wave_paper_autograd.py
    ```
*   Wave equation (initial-boundary value problem)

    ```sh
    python example_wave_physics.py
    ```
*   Heat equation

    ```sh
    python example_heat.py
    ```
*   KdV equation (non-physical conditions for equation discovery problem)

    ```sh
    python example_KdV.py
    ```
*   KdV equation (solitary solution with periodic boundary conditions)

    ```sh
    python example_KdV_periodic.py
    ```
*   Burgers equation and DeepXDE comparison

    ```sh
    python example_Burgers_paper.py
    ```

Examples of how this should work and how it should be used are available [here](https://github.com/ITMO-NSS-team/torch_DE_solver/tree/main/docs/source/tedeous/examples).

---

## Project Structure

Stable version is located in the master branch.

---

## Documentation

A detailed torch_DE_solver description is available [here](https://torch-de-solver.readthedocs.io).

---

## License

This project is protected under the BSD 3-Clause "New" or "Revised" License. For more details, refer to the [LICENSE](https://github.com/ITMO-NSS-team/torch_DE_solver/tree/main/LICENCE) file.

---

## Citation

@article{hvatov2023solver,
  AUTHOR = {Hvatov, Alexander},
  TITLE = {Automated Differential Equation Solver Based on the Parametric Approximation Optimization},
  JOURNAL = {Mathematics},
  VOLUME = {11},
  YEAR = {2023},
  NUMBER = {8},
  ARTICLE-NUMBER = {1787},
  URL = {https://www.mdpi.com/2227-7390/11/8/1787},
  ISSN = {2227-7390},
  DOI = {10.3390/math11081787}
  }