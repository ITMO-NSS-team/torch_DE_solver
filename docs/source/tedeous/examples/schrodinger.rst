Schrodinger equation
====================

Problem statement
~~~~~~~~~~~~~~~~~
Equation:

.. math:: i \frac{\partial u}{\partial t} + \frac{1}{2} \frac{\partial^2 u}{\partial x^2} + \lvert h^2 \rvert h = 0, \qquad x \in [-5,5], \quad t \in [0, \pi/2]

Boundary and initial condition:

.. math:: h(t, -5) = h(t, 5), \quad h_x(t, -5) = h_x(t,5), \quad h(0,x) = 2 sech(x)

Solution
~~~~~~~~
First of all import all dependencies.

.. code-block:: python
	
	import numpy as np
	import torch

	from tedeous.solver import Solver
	from tedeous.input_preprocessing import Equation

After that let's define a computational grid.

.. code-block:: python

	x_grid = np.linspace(-5,5,n+1)
	t_grid = np.linspace(0,np.pi/2,n+1)
	x = torch.from_numpy(x_grid)
	t = torch.from_numpy(t_grid)
	grid = torch.cartesian_prod(x, t).float()
	grid.to(device)

Now let's define the boundary and initial conditions.

.. code-block:: python

	fun = lambda x: 2/np.cosh(x)
						
    	# u(x,0) = 2sech(x), v(x,0) = 0
   	bnd1_real = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
    	bnd1_imag = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()


    	# u(x,0) = 2sech(x)
    	bndval1_real = fun(bnd1_real[:,0])

	#  v(x,0) = 0
	bndval1_imag = torch.from_numpy(np.zeros_like(bnd1_imag[:,0]))


    	# u(-5,t) = u(5,t)
    	bnd2_real_left = torch.cartesian_prod(torch.from_numpy(np.array([-5], dtype=np.float64)), t).float()
    	bnd2_real_right = torch.cartesian_prod(torch.from_numpy(np.array([5], dtype=np.float64)), t).float()
    	bnd2_real = [bnd2_real_left,bnd2_real_right]

    	# v(-5,t) = v(5,t)
    	bnd2_imag_left = torch.cartesian_prod(torch.from_numpy(np.array([-5], dtype=np.float64)), t).float()
    	bnd2_imag_right = torch.cartesian_prod(torch.from_numpy(np.array([5], dtype=np.float64)), t).float()
  	bnd2_imag = [bnd2_imag_left,bnd2_imag_right]


    	# du/dx (-5,t) = du/dx (5,t)
    	bnd3_real_left = torch.cartesian_prod(torch.from_numpy(np.array([-5], dtype=np.float64)), t).float()
    	bnd3_real_right = torch.cartesian_prod(torch.from_numpy(np.array([5], dtype=np.float64)), t).float()
    	bnd3_real = [bnd3_real_left, bnd3_real_right]

    	bop3_real = {
    		    'du/dx':
                    {
                        'coeff': 1,
                        'du/dx': [0],
                        'pow': 1,
                        'var': 0
                    }
    	}
    	# dv/dx (-5,t) = dv/dx (5,t)
   	bnd3_imag_left = torch.cartesian_prod(torch.from_numpy(np.array([-5], dtype=np.float64)), t).float()
    	bnd3_imag_right = torch.cartesian_prod(torch.from_numpy(np.array([5], dtype=np.float64)), t).float()
    	bnd3_imag = [bnd3_imag_left,bnd3_imag_right]

    	bop3_imag = {
                'dv/dx':
                    {
                        'coeff': 1,
                        'dv/dx': [0],
                        'pow': 1,
                        'var': 1
                    }
    	}


    	bcond_type = 'periodic'

    	bconds = [[bnd1_real, bndval1_real, 0],
             	 [bnd1_imag, bndval1_imag, 1],
             	 [bnd2_real, 0, bcond_type],
             	 [bnd2_imag, 1, bcond_type],
             	 [bnd3_real, bop3_real, bcond_type],
             	 [bnd3_imag, bop3_imag, bcond_type]]

Now define the equation.

.. code-block:: python

	schrodinger_eq_real = {
            'du/dt':
                {
                    'const': 1,
                    'term': [1],
                    'power': 1,
                    'var': 0
                },
            '1/2*d2v/dx2':
                {
                    'const': 1 / 2,
                    'term': [0, 0],
                    'power': 1,
                    'var': 1
                },
            'v * u**2':
                {
                    'const': 1,
                    'term': [[None], [None]],
                    'power': [1, 2],
                    'var': [1, 0]
                },
            'v**3':
                {
                    'const': 1,
                    'term': [None],
                    'power': 3,
                    'var': 1
                }

        }
    	schrodinger_eq_imag = {
            'dv/dt':
                {
                    'const': 1,
                    'term': [1],
                    'power': 1,
                    'var': 1
                },
            '-1/2*d2u/dx2':
                {
                    'const': - 1 / 2,
                    'term': [0, 0],
                    'power': 1,
                    'var': 0
                },
            '-u * v ** 2':
                {
                    'const': -1,
                    'term': [[None], [None]],
                    'power': [1, 2],
                    'var': [0, 1]
                },
            '-u ** 3':
                {
                    'const': -1,
                    'term': [None],
                    'power': 3,
                    'var': 0
                }

        	}

    	schrodinger_eq = [schrodinger_eq_real,schrodinger_eq_imag]

Initialize the model.

.. code-block:: python

	 model = torch.nn.Sequential(
                torch.nn.Linear(2, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 2)
            )

Wrap grid, equation, conditions in one entity. This step requires to specify a calculation strategy.

.. code-block:: python

	equation = Equation(grid, schrodinger_eq, bconds).set_strategy('autograd')

And in the end you have to apply all these stuff in Solver class.

.. code-block:: python

	 model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=1, verbose=True, learning_rate=0.8,
                                            eps=1e-6, tmin=1000, tmax=1e5,use_cache=True,cache_dir='../cache/',cache_verbose=True,
                                            save_always=False,no_improvement_patience=500,print_every = None, optimizer_mode='LBFGS', step_plot_print=False, 						    step_plot_save=True, image_save_dir=img_dir)
