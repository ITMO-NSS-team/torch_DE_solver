import pandas as pd
import numpy as np
import matplotlib
import epde.interface.interface as epde_alg
import epde.globals as global_var
from epde.evaluators import CustomEvaluator
import os
from epde.interface.prepared_tokens import Custom_tokens, Trigonometric_tokens, Cache_stored_tokens

import matplotlib.pyplot as plt
import torch
import TEDEouS

from TEDEouS import solver
from TEDEouS import config

os.chdir(os.path.dirname(__file__))


def load_data(mesh):
    """
        Load data
        Synthetic data from wolfram:

        WE = {D[u[x, t], {t, 2}] - 1/25 ( D[u[x, t], {x, 2}]) == 0}
        bc = {u[0, t] == 0, u[1, t] == 0};
        ic = {u[x, 0] == 10000 Sin[1/10 x (x - 1)]^2, Evaluate[D[u[x, t], t] /. t -> 0] == 1000 Sin[1/10  x (x - 1)]^2}
        NDSolve[Flatten[{WE, bc, ic}], u, {x, 0, 1}, {t, 0, 1}]
    """

    df = pd.read_csv(f'wolfram_sln/wave_sln_{mesh}.csv', header=None)
    u = df.values
    u = np.transpose(u)  # x1 - t (axis Y), x2 - x (axis X)

    t = np.linspace(0, 1, mesh + 1)
    x = np.linspace(0, 1, mesh + 1)
    grid = np.meshgrid(t, x, indexing='ij')
    params = [t, x]

    return u, grid, params


def equation_fit(grid, data):
    boundary = 0
    dimensionality = data.ndim

    epde_search_obj = epde_alg.epde_search(use_solver=False, eq_search_iter=100,
                                           dimensionality=dimensionality)

    epde_search_obj.set_memory_properties(data, mem_for_cache_frac=10)
    epde_search_obj.set_moeadd_params(population_size=10, training_epochs=5)

    custom_grid_tokens = Cache_stored_tokens(token_type='grid',
                                             boundary=boundary,
                                             token_labels=['t', 'x'],
                                             token_tensors={'t': grid[0], 'x': grid[1]},
                                             params_ranges={'power': (1, 1)},
                                             params_equality_ranges=None)

    epde_search_obj.fit(data=data, max_deriv_order=(2, 2), boundary=boundary,
                        equation_terms_max_number=3, equation_factors_max_number=1,
                        coordinate_tensors=grid, eq_sparsity_interval=(1e-8, 5.0),
                        deriv_method='poly', deriv_method_kwargs={'smooth': True, 'grid': grid},
                        additional_tokens=[custom_grid_tokens, ],
                        memory_for_cache=25, prune_domain=False)

    res = epde_search_obj.equation_search_results(only_print=False, level_num=2)

    solver_inp = []

    for eq in res[0]:
        solver_inp.append((eq.structure[0].solver_form(), eq.structure[0].boundary_conditions()))

    epde_search_obj.equation_search_results(only_print=True, level_num=1)
    return solver_inp


if __name__ == '__main__':

    grid_res = 50

    u, grid_u, params = load_data(grid_res)

    solver_inp = equation_fit(grid_u, u)

    models = []

    for eqn_n, s_inp in enumerate(solver_inp):
        coord_list = params

        cfg = config.Config()

        cfg.set_parameter("Cache.use_cache", True)
        cfg.set_parameter('Cache.save_always', True)

        principal_bcond_shape=s_inp[1][0][1].shape

        for i in range(len(s_inp[1])):
            s_inp[1][i][1]=s_inp[1][i][1].reshape(principal_bcond_shape)



        model = solver.optimization_solver(coord_list, None, s_inp[0], s_inp[1], cfg, mode='NN')
        print(model)

        models.append(model)






