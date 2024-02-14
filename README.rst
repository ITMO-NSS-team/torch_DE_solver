TEDEouS - Torch Exhaustive Differential Equation Solver
=======================================================



The purpose of the project
--------------------------

1. Make equation discovery more transparent and illustrative
2. Combine power of pytorch, numerical methods and math overall to conquer and solve ALL XDEs(X={O,P}). There are some examples to provide a little insight to an operator form

Table of Contents
--------------------

- `Core features <Core features_>`_
- `Installation <Installation_>`_
- `Examples <Examples_>`_
- `Project Structure <Project Structure_>`_
- `Documentation <Documentation_>`_
- `Getting started <Getting started_>`_
- `License <License_>`_
- `Contacts <Contacts_>`_
- `Citation <Citation_>`_


Core features
-------------

* Solve ODE initial- or boundary-value problems
* Solve PDE initial-boundary value problems
* Use variable models and their differentiation methods
* Faster solution using cache



Installation
------------

TEDEouS can be installed with ``pip``::

$ git clone https://github.com/ITMO-NSS-team/torch_DE_solver.git
$ cd torch_DE_solver
$ pip install -r requirements.txt


Examples
------------
After the TEDEouS is installed the user may refer to various examples that are in examples forlder. ::

$ cd examples

Every example is designed such that the boxplots of the launches are commented and the preliminary results are not shown, but stored in separate folders.

* Legendre polynomial equation 

::

$ python example_ODE_Legendre.py

or ::

$ python example_ODE_Legendre_autograd.py

* Panleve transcendents (others are placed in 'examples\\to_renew' folder due to the architecture change)

::

$ python example_Painleve_I.py

* Wave equation (non-physical conditions for equation discovery problem) 

::

$ python example_wave_paper_autograd.py

* Wave equation (initial-boundary value problem) 

::

$ python example_wave_physics.py

* Heat equation 

::

$ python example_heat.py

* KdV equation (non-physical conditions for equation discovery problem) 

::

$ python example_KdV.py

* KdV equation (solitary solution with periodic boundary conditions) 

::

$ python example_KdV_periodic.py

* Burgers equation and DeepXDE comparison 

::

$ python example_Burgers_paper.py


Project Structure
-----------------
Stable version is located in the master branch.


Documentation
-------------
https://torch-de-solver.readthedocs.io/en/docs/index.html

Getting started
---------------
Schroedinger equation example step-by-step https://torch-de-solver.readthedocs.io/en/docs/tedeous/examples/schrodinger.html 

License
-------
TEDEouS is distributed under BSD-3 licence found in LICENCE file


Contacts
--------
- Feel free to make issues or contact @SuperSashka directly

Citation
--------

::

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


