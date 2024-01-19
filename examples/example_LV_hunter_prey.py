import time
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.models import FourierNN


x0 = 30.
y0 = 4.

solver_device('cuda')

domain = Domain()
domain.variable('t', [0, 20], 1001)

#initial conditions
boundaries = Conditions()
boundaries.dirichlet({'t': 0}, value=x0, var=0)
boundaries.dirichlet({'t': 0}, value=y0, var=1)

#equation system
# eq1: dx/dt = x(alpha-beta*y)
# eq2: dy/dt = y(-delta+gamma*x)

# x var: 0
# y var:1

equation = Equation()

eq1 = {
    '1/x*dx/dt':{
        'coeff': 1,
        'term': [[None], [0]],
        'pow': [-1, 1],
        'var': [0, 0]
    },
    '-alpha':{
        'coeff': -0.55,
        'term': [None],
        'pow': 0,
        'var': [0]
    },
    '+beta*y':{
        'coeff': 0.028,
        'term': [None],
        'pow': 1,
        'var': [1]
    }
}

eq2 = {
    '1/y*dy/dt':{
        'coeff': 1,
        'term': [[None], [0]],
        'pow': [-1, 1],
        'var': [1, 1]
    },
    '+delta':{
        'coeff': 0.84,
        'term': [None],
        'pow': 0,
        'var': [1]
    },
    '-gamma*x':{
        'coeff': -0.026,
        'term': [None],
        'pow': 1,
        'var': [0]
    }
}

equation.add(eq1)
equation.add(eq2)

net = FourierNN([512, 512, 512, 512, 2], [15], [7])

model =  Model(net, domain, equation, boundaries)

model.compile("NN", lambda_operator=1, lambda_bound=10, tol=0.01)

img_dir=os.path.join(os.path.dirname( __file__ ), 'LV_hunter_prey')

start = time.time()

cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                    loss_window=100,
                                    no_improvement_patience=100,
                                    patience=5,
                                    randomize_parameter=1e-5,
                                    info_string_every=500)

cb_plots = plot.Plots(save_every=500, print_every=None, img_dir=img_dir)

optimizer = Optimizer('Adam', {'lr': 1e-3})

model.train(optimizer, 4e4, save_model=False, callbacks=[cb_cache, cb_es, cb_plots])

end = time.time()