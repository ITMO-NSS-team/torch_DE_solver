# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:50:12 2021

@author: user
"""

import datetime
import os
import glob
import shutil
from copy import deepcopy
from typing import Union, Tuple, Any
import torch
import numpy as np

from tedeous.solution import Solution
from tedeous.input_preprocessing import Operator_bcond_preproc
from tedeous.device import device_type, check_device
from tedeous.callbacks.callback import Callback
from tedeous.utils import create_random_fn
from tedeous.model import Model

def count_output(model: torch.Tensor) -> int:
    """ Determine the out features of the model.

    Args:
        model (torch.Tensor): torch neural network.

    Returns:
        int: number of out features.
    """
    modules, output_layer = list(model.modules()), None
    for layer in reversed(modules):
        if hasattr(layer, 'out_features'):
            output_layer = layer.out_features
            break
    return output_layer


def remove_all_files(folder: str) -> None:
    """ Remove all files from folder.

    Args:
        folder (str): folder name.
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


class CacheUtils:
    """ Mixin class with auxiliary methods
    """
    def __init__(self):
        try:
            file = __file__
        except:
            file = os.getcwd()

        self._cache_dir = os.path.normpath((os.path.join(os.path.dirname(file), '..', '..', 'cache')))

    def get_cache_dir(self):
        """Get cache dir.

        Returns:
            str: cache folder directory.
        """
        return self._cache_dir

    def set_cache_dir(self, string: str) -> None:
        """ Change the directory of cache.

        Args:
            string (str): new cache directory.
        """
        self._cache_dir = string

    def clear_cache_dir(self, directory: Union[str, None] = None) -> None:
        """ Clear cache directory.

        Args:
            directory (str, optional): custom cache directory. Defaults to None.
        """
        if directory is None:
            remove_all_files(self.cache_dir)
        else:
            remove_all_files(directory)

    cache_dir = property(get_cache_dir, set_cache_dir, clear_cache_dir)

    @staticmethod
    def grid_model_mat(model: torch.Tensor,
                       grid: torch.Tensor,
                       cache_model: torch.nn.Module=None) -> Tuple[torch.Tensor, torch.nn.Module]:
        """ Create grid and model for *NN or autograd* modes from grid
            and model of *mat* mode.

        Args:
            model (torch.Tensor): model from *mat* method.
            grid (torch.Tensor): grid from *mat* method.
            cache_model (torch.nn.Module, optional): neural network that will 
                                                     approximate *mat* model. Defaults to None.

        Returns:
            nn_grid (torch.Tensor): grid satisfying neural network inputs.
            cache_model (torch.nn.Module): model satisfying the *NN, autograd* methods.
        """
        nn_grid = torch.vstack([grid[i].reshape(-1) for i in \
                                range(grid.shape[0])]).T.float()
        input_model = grid.shape[0]
        output_model = model.shape[0]

        if cache_model is None:
            cache_model = torch.nn.Sequential(
                torch.nn.Linear(input_model, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, output_model)
            )

        return nn_grid, cache_model

    @staticmethod
    def mat_op_coeff(operator: dict) -> dict:
        """ Preparation of coefficients in the operator of the *mat* method
            to suit methods *NN, autograd*.

        Args:
            operator (dict): operator (equation dict).

        Returns:
            operator (dict): operator (equation dict) with suitable coefficients.
        """
        if not isinstance(operator, list):
            operator = [operator]
        for op in operator:
            for label in list(op.keys()):
                term = op[label]
                if isinstance(term['coeff'], torch.Tensor):
                    term['coeff'] = term['coeff'].reshape(-1, 1)
                elif callable(term['coeff']):
                    print("Warning: coefficient is callable,\
                                    it may lead to wrong cache item choice")
        return operator

    def save_model(
        self,
        model: torch.nn.Module,
        name: Union[str, None] = None) -> None:
        """
        Saved model in a cache (uses for 'NN' and 'autograd' methods).
        Args:
            model (torch.nn.Module): model to save.
            (uses only with mixed precision and device=cuda). Defaults to None.
            name (str, optional): name for a model. Defaults to None.
        """

        if name is None:
            name = str(datetime.datetime.now().timestamp())
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)
        parameters_dict = {'model': model.to('cpu'),
                           'model_state_dict': model.state_dict()}

        try:
            torch.save(parameters_dict, self.cache_dir + '\\' + name + '.tar')
            print('model is saved in cache')
        except RuntimeError:
            torch.save(parameters_dict, self.cache_dir + '\\' + name + '.tar',
                       _use_new_zipfile_serialization=False)  # cyrrilic in path
            print('model is saved in cache')
        except:
            print('Cannot save model in cache')

    def save_model_mat(self, model: torch.Tensor,
                       grid: torch.Tensor,
                       cache_model: Union[torch.nn.Module, None] = None,
                       name: Union[str, None] = None) -> None:
        """ Saved model in a cache (uses for 'mat' method).

        Args:
            model (torch.Tensor): *mat* model
            grid (torch.Tensor): grid from *mat* mode
            cache_model (Union[torch.nn.Module, None], optional): model to save. Defaults to None.
            name (Union[str, None], optional): name for a model. Defaults to None.
        """

        nn_grid, cache_model = self.grid_model_mat(model, grid, cache_model)
        optimizer = torch.optim.Adam(cache_model.parameters(), lr=0.001)
        model_res = model.reshape(-1, model.shape[0])

        def closure():
            optimizer.zero_grad()
            loss = torch.mean((cache_model(check_device(nn_grid)) - model_res) ** 2)
            loss.backward()
            return loss

        loss = np.inf
        t = 0
        while loss > 1e-5 and t < 1e5:
            loss = optimizer.step(closure)
            t += 1
            print('Interpolate from trained model t={}, loss={}'.format(
                    t, loss))

        self.save_model(cache_model, name=name)


class CachePreprocessing:
    """class for preprocessing cache files.
    """
    def __init__(self,
                 model: Model
                 ):
        """
        Args:
            grid (torch.Tensor): grid (domain discretization)
            equal_cls (Equation class object): Equation class object that contain preprocessed
                                               operator (equation) and boundary con-s.
            model (Union[torch.Tensor, torch.nn.Module]): model (neural network or tensor)
            mode (str): solution strategy (mat, NN, autograd)
            weak_form (Union[list, None]): list that contains basis function for weak solution.
                                           If None: soluion in strong form.
            mixed_precision (bool): flag for on/off torch.amp.
        """
        self.solution_cls = model.solution_cls

    @staticmethod
    def _cache_files(files: list, nmodels: Union[int, None]=None) -> np.ndarray:
        """ At some point we may want to reduce the number of models that are
            checked for the best in the cache.

        Args:
            files (list): list with all model names in cache.
            nmodels (Union[int, None], optional): models quantity for checking. Defaults to None.

        Returns:
            cache_n (np.ndarray): array with random cache files names.
        """

        if nmodels is None:
            # here we take all files that are in cache
            cache_n = np.arange(len(files))
        else:
            # here we take random nmodels from the cache
            cache_n = np.random.choice(len(files), nmodels, replace=False)

        return cache_n

    @staticmethod
    def _model_reform(init_model: Union[torch.nn.Sequential, torch.nn.ModuleList],
                     model: Union[torch.nn.Sequential, torch.nn.ModuleList]):
        """
        As some models are nn.Sequential class objects,
        but another models are nn.Module class objects.
        This method does checking the solver model (init_model)
        and the cache model (model).
        Args:
            init_model (nn.Sequential or nn.ModuleList): solver model.
            model (nn.Sequential or nn.ModuleList): cache model.
        Returns:
            init_model (nn.Sequential or nn.ModuleList): checked init_model.
            model (nn.Sequential or nn.ModuleList): checked model.
        """
        try:
            model[0]
        except:
            model = model.model

        try:
            init_model[0]
        except:
            init_model = init_model.model

        return init_model, model

    def cache_lookup(self,
                     nmodels: Union[int, None] = None,
                     save_graph: bool = False,
                     cache_verbose: bool = False) -> Union[None, dict, torch.nn.Module]:
        """Looking for the best model (min loss) model from the cache files.

        Args:
            lambda_operator (float, optional): regulariazation parameter for operator term in loss. Defaults to 1.
            lambda_bound (float, optional): regulariazation parameter for boundary term in loss. Defaults to 1.
            nmodels (Union[int, None], optional): maximal number of models that are taken from cache dir. Defaults to None.
            save_graph (bool, optional): responsible for saving the computational graph. Defaults to False.
            cache_verbose (bool, optional): verbose cache operations. Defaults to False.

        Returns:
            Union[None, dict, torch.Tensor]: best model with optimizator state.
        """

        files = glob.glob(CacheUtils().cache_dir + '\*.tar')

        if len(files) == 0:
            best_checkpoint = None
            return best_checkpoint

        cache_n = self._cache_files(files, nmodels)
        
        min_loss = np.inf
        best_checkpoint = {}

        device = device_type()

        initial_model = self.solution_cls.model

        for i in cache_n:
            file = files[i]
            checkpoint = torch.load(file)

            model = checkpoint['model']
            model.load_state_dict(checkpoint['model_state_dict'])

            # this one for the input shape fix if needed

            solver_model, cache_model = self._model_reform(self.solution_cls.model, model)

            if cache_model[0].in_features != solver_model[0].in_features:
                continue
            try:
                if count_output(model) != count_output(self.solution_cls.model):
                    continue
            except Exception:
                continue

            model = model.to(device)
            self.solution_cls.model = model
            loss, _ = self.solution_cls.evaluate(save_graph=save_graph)

            if loss < min_loss:
                min_loss = loss
                best_checkpoint['model'] = model
                best_checkpoint['model_state_dict'] = model.state_dict()
                if cache_verbose:
                    print('best_model_num={} , loss={}'.format(i, min_loss.item()))

            self.solution_cls.model = initial_model

        if best_checkpoint == {}:
            best_checkpoint = None

        return best_checkpoint

    def scheme_interp(self,
                      trained_model: torch.nn.Module,
                      cache_verbose: bool = False) -> torch.nn.Module:
        """ If the cache model has another arcitechure to user's model,
            we will not be able to use it. So we train user's model on the
            outputs of cache model.

        Args:
            trained_model (torch.nn.Module): the best model (min loss) from cache.
            cache_verbose (bool, optional): verbose on/off of cache operations. Defaults to False.

        Returns:
            self.model (torch.nn.Module): model trained on the cache model outputs.

        """

        grid = self.solution_cls.grid

        model = self.solution_cls.model

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss = torch.mean(torch.square(
            trained_model(grid) - model(grid)))

        def closure():
            optimizer.zero_grad()
            loss = torch.mean((trained_model(grid) - model(grid)) ** 2)
            loss.backward()
            return loss

        t = 0
        while loss > 1e-5 and t < 1e5:
            optimizer.step(closure)
            loss = torch.mean(torch.square(
                trained_model(grid) - model(grid)))
            t += 1
            if cache_verbose:
                print('Interpolate from trained model t={}, loss={}'.format(
                    t, loss))
        
        self.solution_cls.model = model

    def cache_retrain(self,
                      cache_checkpoint: dict,
                      cache_verbose: bool = False) -> torch.nn.Module:
        """ The comparison of the user's model and cache model architecture.
            If they are same, we will use model from cache. In the other case
            we use interpolation (scheme_interp method)

        Args:
            cache_checkpoint (dict): checkpoint of the cache model
            cache_verbose (bool, optional): on/off printing cache operations. Defaults to False.

        Returns:
            model (torch.nn.Module): the resulting model.
            optimizer_state (dict): the state of the optimizer.
        """

        model = self.solution_cls.model

        # do nothing if cache is empty
        if cache_checkpoint is None:
            return None
        # if models have the same structure use the cache model state,
        # and the cache model has ordinary structure
        if str(cache_checkpoint['model']) == str(model) and \
                isinstance(model, torch.nn.Sequential) and \
                isinstance(model[0], torch.nn.Linear):
            model = cache_checkpoint['model']
            model.load_state_dict(cache_checkpoint['model_state_dict'])
            model.train()
            self.solution_cls.model = model
            if cache_verbose:
                print('Using model from cache')
        # else retrain the input model using the cache model
        else:
            cache_model = cache_checkpoint['model']
            cache_model.load_state_dict(cache_checkpoint['model_state_dict'])
            cache_model.eval()
            self.scheme_interp(
                cache_model, cache_verbose=cache_verbose)


class Cache(Callback):
    """
    Prepares user's model. Serves for computing acceleration.\n
    Saves the trained model to the cache, and subsequently it is possible to use pre-trained model
    (if it saved and if the new model is structurally similar) to sped up computing.\n
    If there isn't pre-trained model in cache, the training process will start from the beginning.
    """

    def __init__(self,
                 nmodels: Union[int, None] = None,
                 cache_dir: str = '../cache/',
                 cache_verbose: bool = False,
                 cache_model: Union[torch.nn.Sequential, None] = None,
                 model_randomize_parameter: Union[int, float] = 0,
                 clear_cache: bool = False
                ):
        """
        Args:
            grid (torch.Tensor): grid (domain discretization)
            equal_cls (Equation class object): Equation class object that contain preprocessed
                                               operator (equation) and boundary con-s.
            model (Union[torch.Tensor, torch.nn.Module]): model (neural network or tensor)
            mode (str): solution strategy (mat, NN, autograd)
            weak_form (Union[list, None]): list that contains basis function for weak solution.
                                           If None: soluion in strong form.
            mixed_precision (bool): flag for on/off torch.amp.
        """
        self.nmodels = nmodels
        self.cache_dir = cache_dir
        self.cache_verbose = cache_verbose
        self.cache_model = cache_model
        self.model_randomize_parameter = model_randomize_parameter
        self.clear_cache = clear_cache

    def _cache_nn(self):
        """  Restores the model from the cache and uses it for *NN, autograd* modes.

        Args:
            nmodels (Union[int, None]): model quantity taken from cache directory.
            lambda_operator (float): regulariazation parameter for operator term in loss.
            lambda_bound (float): regulariazation parameter for boundary term in loss.
            cache_verbose (bool): verbose cache operations.
            model_randomize_parameter (Union[float, None]): some error for resulting
            model weights to to avoid local optima.

        Returns:
            model (torch.nn.Module): final model for optimization
        """
        cache_preproc = CachePreprocessing(self.model)

        r = create_random_fn(self.model_randomize_parameter)

        cache_checkpoint = cache_preproc.cache_lookup(nmodels=self.nmodels,
                                                    cache_verbose=self.cache_verbose)

        cache_preproc.cache_retrain(cache_checkpoint,
                                               cache_verbose=self.cache_verbose)
        self.model.solution_cls.model.apply(r)

    def _cache_mat(self) -> torch.Tensor:
        """ Restores the model from the cache and uses it for *mat* mode.

        Args:
            nmodels (Union[int, None]): model quantity taken from cache directory.
            lambda_operator (float):regulariazation parameter for operator term in loss.
            lambda_bound (float): regulariazation parameter for boundary term in loss.
            cache_verbose (bool): verbose cache operations.
            model_randomize_parameter (Union[float, None]): some error for resulting
            model weights to to avoid local optima.
            cache_model (Union[torch.nn.Module, None]): user defined cache model.

        Returns:
            model (torch.Tensor): resulting model for *mat* mode.
        """

        net = self.model.net
        domain = self.model.domain
        equation = self.model.equation
        conditions = self.model.conditions
        lambda_operator = self.model.lambda_operator
        lambda_bound = self.model.lambda_bound
        weak_form = self.model.weak_form

        autograd_model = Model(net, domain, equation, conditions)

        autograd_model.compile('autograd', lambda_operator, lambda_bound, weak_form=weak_form)

        r = create_random_fn(self.model_randomize_parameter)

        cache_preproc = CachePreprocessing(autograd_model)

        cache_checkpoint = cache_preproc.cache_lookup(
            nmodels=self.nmodels,
            cache_verbose=self.cache_verbose)

        if cache_checkpoint is not None:
            cache_preproc.cache_retrain(
                cache_checkpoint,
                cache_verbose=self.cache_verbose)

            autograd_model.solution_cls.model.apply(r)

            model = autograd_model.solution_cls.model(
                autograd_model.solution_cls.grid).reshape(
                self.model.solution_cls.model.shape).detach()
            
            self.model.solution_cls.model = model

    def cache(self):
        """ Wrap for cache_mat and cache_nn methods.

        Args:
            nmodels (Union[int, None]): model quantity taken from cache directory.
            lambda_operator (_type_): regulariazation parameter for operator term in loss.
            lambda_bound (float): regulariazation parameter for boundary term in loss.
            cache_verbose (bool):  verbose cache operations.
            model_randomize_parameter (Union[float, None]): some error for resulting
            model weights to to avoid local optima.
            cache_model (Union[torch.nn.Module, None]): user defined cache model.

        Returns:
            cache.cache_nn or cache.cache_mat
        """

        if self.model.mode != 'mat':
            return self._cache_nn()
        elif self.mdoel.mode == 'mat':
            return self._cache_mat()

    def on_train_begin(self, logs=None):
        self.cache()
