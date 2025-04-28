import time
import torch
import os
import sys
torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data
import wandb

wandb.login()

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    # Set the wandb project where this run will be logged.
    project="rlpinn",
    # Track hyperparameters and run metadata.
    config={
        "param": "v_1",
        "reward_function": "v_1",
        "buffer_size": 4,
        "batch_size": 2,
        "type_buffer": "partly_minus_butch_size"
    },
)

# solver_device('cuda')
solver_device('cpu')
# torch.set_default_device("cpu")
# torch.set_default_device('mps:0')

datapath = "../../PINNacle_data/poisson1_cg_data.npy"


def poisson_2d_classic_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = -0.5, 0.5
    y_min, y_max = -0.5, 0.5

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)

    boundaries = Conditions()

    # Circle type of removed domains ###################################################################################

    removed_domains_lst = [
        {'circle': {'center': (0.3, 0.3), 'radius': 0.1}},
        {'circle': {'center': (-0.3, 0.3), 'radius': 0.1}},
        {'circle': {'center': (0.3, -0.3), 'radius': 0.1}},
        {'circle': {'center': (-0.3, -0.3), 'radius': 0.1}}
    ]

    # Boundary conditions ##############################################################################################

    # CSG boundaries

    boundaries.dirichlet({'circle': {'center': (0.3, 0.3), 'radius': 0.1}}, value=0)
    boundaries.dirichlet({'circle': {'center': (-0.3, 0.3), 'radius': 0.1}}, value=0)
    boundaries.dirichlet({'circle': {'center': (0.3, -0.3), 'radius': 0.1}}, value=0)
    boundaries.dirichlet({'circle': {'center': (-0.3, -0.3), 'radius': 0.1}}, value=0)

    # Non CSG boundaries

    boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max]}, value=1)
    boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max]}, value=1)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min}, value=1)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max}, value=1)

    equation = Equation()

    # Operator: -d2u/dx2 - d2u/dy2 = 0

    poisson = {
        '-d2u/dx2':
            {
                'coeff': -1.,
                'term': [0, 0],
                'pow': 1,
                'var': 0
            },
        '-d2u/dy2':
            {
                'coeff': -1.,
                'term': [1, 1],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(poisson)

    neurons = 100

    net = torch.nn.Sequential(
        torch.nn.Linear(pde_dim_in, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, pde_dim_out)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    start = time.time()

    grid = domain.build('autograd')
    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))

    t_dim_flag = 't' in list(domain.variable_dict.keys())

    grid_params = domain.build('NN').to('cuda')
    exact = exact_solution_data(grid_params, datapath, pde_dim_in, pde_dim_out,
                                t_dim_flag=t_dim_flag).reshape(-1, 1)

    model_layers = [pde_dim_in, neurons, neurons, neurons, neurons, pde_dim_out]

    equation_params = [exact, grid_test, grid, domain, equation, boundaries, model_layers]

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100, removed_domains=removed_domains_lst)

    img_dir = os.path.join(os.path.dirname(__file__), 'poisson_2d_classic_img')

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=100,
                                         patience=20,
                                         randomize_parameter=1e-4,
                                         info_string_every=10)

    cb_plots = plot.Plots(save_every=None,
                          print_every=None,
                          img_dir=img_dir,
                          scatter_flag=False)

    # RL wrapper integration ###########################################################################################
    # The wrapper should be inside the model.train method
    # We need to change the optimizers inside the RL learning loop and substitute them into the model learning loop
    # We can change optimizers only inside model.train and inside the RL loop
    # Other optimizer replacement configurations (e.g., within the example equation) will cause the training loop to not
    # work correctly due to the update of the model.train method

    # Following parameters must be entered into the model
    # these parameters took from /landscape_visualization_origin/_aux/plot_loss_surface.py file:

    # optimizer = {
    #     'CSO': ({"lr": 1e-3}, 30),
    #     'Adam': ({"lr": 1e-4}, 70),
    #     'LBFGS': ({"lr": 1,
    #                "max_iter": 20,
    #                "max_eval": None,
    #                "tolerance_grad": 1e-05,
    #                "tolerance_change": 1e-07,
    #                "history_size": 50,
    #                "line_search_fn": "strong_wolfe"}, 50),
    #     # 'NGD': ({'grid_steps_number': 20}, 30),
    #     'NNCG': ({"mu": 1e-1,
    #               "lr": 1,
    #               "rank": 10,
    #               "line_search_fn": "armijo",
    #               "precond_update_frequency": 20,
    #               "eigencdecomp_shift_attepmt_count": 10,
    #               # 'cg_max_iters': 1000,
    #               "verbose": False}, 50)
    # }

    # optimizer = [
    #     # {
    #     #     "name": "CSO",
    #     #     "params": {"lr": 5e-4},
    #     #     "epochs": 100
    #     # },
    #     {
    #         "name": "Adam",
    #         "params": {"lr": 1e-4},
    #         "epochs": 50
    #     },
    #     # {
    #     #     "name": "LBFGS",
    #     #     "params": {
    #     #         "lr": 1e-1,
    #     #         "max_iter": 20,
    #     #         "max_eval": None,
    #     #         "tolerance_grad": 1e-05,
    #     #         "tolerance_change": 1e-07,
    #     #         "history_size": 50,
    #     #         "line_search_fn": "strong_wolfe"
    #     #     }, "epochs": 100
    #     # },
    #     {
    #         "name": "NNCG",
    #         "params": {
    #             "mu": 1e-1,
    #             "lr": 1,
    #             "rank": 10,
    #             "line_search_fn": "armijo",
    #             "precond_update_frequency": 20,
    #             "eigencdecomp_shift_attepmt_count": 10,
    #             # 'cg_max_iters': 1000,
    #             "verbose": False},
    #         "epochs": 50
    #     }
    # ]

    optimizer = {
        "type": ['Adam', 'RAdam', 'LBFGS', 'PSO', 'CSO', 'RMSprop'],
        "params": [0.1, 0.01, 0.001, 0.0001],
        "epochs": [100, 500, 1000]
    }

    # optimizer = Optimizer('Adam', {'lr': 1e-4})

    AE_model_params = {
        "mode": "NN",
        "num_of_layers": 3,
        "layers_AE": [
            991,
            125,
            15
        ],
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "every_nth": 1,
        "grid_step": 0.1,
        "d_max_latent": 2,
        "anchor_mode": "circle",
        "rec_weight": 10000.0,
        "anchor_weight": 0.0,
        "lastzero_weight": 0.0,
        "polars_weight": 0.0,
        "wellspacedtrajectory_weight": 0.0,
        "gridscaling_weight": 0.0,
        "device": "cpu"
    }

    AE_train_params = {
        "first_RL_epoch_AE_params": {
            "epochs": 10000,
            "patience_scheduler": 4000,
            "cosine_scheduler_patience": 1200,
        },
        "other_RL_epoch_AE_params": {
            "epochs": 20000,
            "patience_scheduler": 4000,
            "cosine_scheduler_patience": 1200,
        },
        "batch_size": 32,
        "every_epoch": 100,
        "learning_rate": 5e-4,
        "resume": True,
        "finetune_AE_model": False
    }

    loss_surface_params = {
        "loss_types": ["loss_total", "loss_oper", "loss_bnd"],
        "every_nth": 1,
        "num_of_layers": 3,
        "layers_AE": [
            991,
            125,
            15
        ],
        "batch_size": 32,
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "loss_name": "loss_total",
        "x_range": [-1.25, 1.25, 25],
        "vmax": -1.0,
        "vmin": -1.0,
        "vlevel": 30.0,
        "key_models": None,
        "key_modelnames": None,
        "density_type": "CKA",
        "density_p": 2,
        "density_vmax": -1,
        "density_vmin": -1,
        "colorFromGridOnly": True,
        "img_dir": img_dir
    }

    rl_agent_params = {
        "n_save_models": 10,
        "n_trajectories": 1000,
        "tolerance": 1e-1,
        "stuck_threshold": 10,  # Число эпох без значительного изменения прогресса
        "rl_buffer_size": 4,
        "rl_batch_size": 32,
        "rl_reward_method": "absolute",
        "exact_solution": datapath,
        "reward_operator_coeff": 1,
        "reward_boundary_coeff": 1
    }

    model.train(optimizer,
                5e5,
                save_model=True,
                callbacks=[cb_es, cb_plots, cb_cache],
                rl_agent_params=rl_agent_params,
                models_concat_flag=False,
                model_name='rl_optimization_agent',
                equation_params=equation_params,
                AE_model_params=AE_model_params,
                AE_train_params=AE_train_params,
                loss_surface_params=loss_surface_params
                )

    end = time.time()

    # grid = domain.build('NN').to('cuda')
    # net = net.to('cuda')

    exact = exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out, t_dim_flag=t_dim_flag).reshape(-1, 1)
    net_predicted = net(grid)

    error_rmse = torch.sqrt(torch.mean((exact - net_predicted) ** 2))

    exp_dict_list.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE': error_rmse.detach().cpu().numpy(),
        'type': 'poisson_2d_classic',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(50, 501, 50):
    for _ in range(nruns):
        exp_dict_list.append(poisson_2d_classic_experiment(grid_res))


import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/poisson_2d_classic_experiment_50_500_cache={}.csv'.format(str(True)))


