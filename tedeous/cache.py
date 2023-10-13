# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:50:12 2021

@author: user
"""
import pickle
import datetime
import torch
import os
import glob
import numpy as np
import shutil
from copy import deepcopy
from typing import Union, Tuple, Any

from tedeous.solution import Solution
from tedeous.input_preprocessing import Equation, EquationMixin
from tedeous.device import device_type


def count_output(model):
    modules, output_layer = list(model.modules()), None
    for layer in reversed(modules):
        if hasattr(layer, 'out_features'):
            output_layer = layer.out_features
            break
    return output_layer


def create_random_fn(eps):
    def randomize_params(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            m.weight.data = m.weight.data + \
                            (2 * torch.randn(m.weight.size()) - 1) * eps
            m.bias.data = m.bias.data + (2 * torch.randn(m.bias.size()) - 1) * eps

    return randomize_params

def remove_all_files(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

class Model_prepare():
    """
    Prepares initial model. Serves for computing acceleration.\n
    Saves the trained model to the cache, and subsequently it is possible to use pre-trained model (if \\\
    it saved and if the new model is structurally similar) to sped up computing.\n
    If there isn't pre-trained model in cache, the training process will start from the beginning.
    """
    def __init__(self, grid, equal_cls, model, mode, weak_form):
        self.grid = grid
        self.equal_cls = equal_cls
        self.model = model
        self.mode = mode
        self.weak_form = weak_form
        try:
             file = __file__
        except:
            file = os.getcwd()
        self.cache_dir = os.path.normpath((os.path.join(os.path.dirname(file), '..','cache')))

    def change_cache_dir(self, string):
        self.cache_dir=string
        return None

    def clear_cache_dir(self, directory=None):
        if directory==None:
            remove_all_files(self.cache_dir)
        else:
            remove_all_files(directory)
        return None

    @staticmethod
    def cache_files(files, nmodels):

        # at some point we may want to reduce the number of models that are
        # checked for the best in the cache
        if nmodels == None:
            # here we take all files that are in cache
            cache_n = np.arange(len(files))
        else:
            # here we take random nmodels from the cache
            cache_n = np.random.choice(len(files), nmodels, replace=False)

        return cache_n

    def grid_model_mat(self, cache_model):
        NN_grid = torch.vstack([self.grid[i].reshape(-1) for i in \
                                range(self.grid.shape[0])]).T.float()
        out = self.model.shape[0]

        if cache_model == None:
            cache_model = torch.nn.Sequential(
                torch.nn.Linear(self.grid.shape[0], 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, out)
            )
        return NN_grid, cache_model

    @staticmethod
    def mat_op_coeff(operator):
        if type(operator) is not list:
            operator = [operator]
        for op in operator:
            for label in list(op.keys()):
                term = op[label]
                if type(term['coeff']) == torch.Tensor:
                    term['coeff'] = term['coeff'].reshape(-1, 1)
                elif callable(term['coeff']):
                    print("Warning: coefficient is callable,\
                                it may lead to wrong cache item choice")
        return operator

    @staticmethod
    def model_reform(init_model, model):
        """
        As some models are nn.Sequential class objects,
        but another models are nn.Module class objects.
        This method does checking the solver model (init_model)
        and the cache model (model).
        Args:
            init_model: [nn.Sequential or class(nn.Module)].
            model: [nn.Sequential or class(nn.Module)].
        Returns:
            * **init_model** -- [nn.Sequential or nn.ModuleList] \n
            * **model** -- [nn.Sequential or nn.ModuleList].
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


    def cache_lookup(self, lambda_operator: float = 1., lambda_bound: float = 0.001,
                nmodels: Union[int, None] = None, save_graph: bool = False,
                cache_verbose: bool = False, return_normalized_loss: bool = False) -> Tuple[dict, torch.Tensor]:
        """
        Looking for a saved cache.
        Args:
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
            save_graph: boolean constant, responsible for saving the computational graph.
            cache_dir: directory where saved cache in.
            nmodels: maximal number of models that are looked before optimization
            cache_verbose: more detailed info about models in cache.
        Returns:
            * **best_checkpoint** -- best model with optimizator state.\n
            * **min_loss** -- minimum error in pre-trained error.
        """
        files = glob.glob(self.cache_dir + '\*.tar')
        if len(files) == 0:
            best_checkpoint = None
            min_loss = torch.tensor([float('inf')])
            return best_checkpoint, min_loss

        cache_n = self.cache_files(files, nmodels)

        min_loss = np.inf
        min_norm_loss =np.inf
        best_checkpoint = {}

        device = device_type()

        for i in cache_n:
            file = files[i]
            checkpoint = torch.load(file)
            model = checkpoint['model']
            model.load_state_dict(checkpoint['model_state_dict'])
            # this one for the input shape fix if needed

            solver_model, cache_model = self.model_reform(self.model, model)

            if cache_model[0].in_features != solver_model[0].in_features:
                continue
            try:
                if count_output(model) != count_output(self.model):
                    continue
            except Exception:
                continue

            model = model.to(device)
            loss, loss_normalized = Solution(self.grid, self.equal_cls,
                                      model, self.mode, self.weak_form,
                                      lambda_operator, lambda_bound, tol=0,
                                      derivative_points=2).evaluate(save_graph=save_graph)

            if loss < min_loss:
                min_loss = loss
                min_norm_loss=loss_normalized
                best_checkpoint['model'] = model
                best_checkpoint['model_state_dict'] = model.state_dict()
                best_checkpoint['optimizer_state_dict'] = \
                    checkpoint['optimizer_state_dict']
                if cache_verbose:
                    print('best_model_num={} , normalized_loss={}'.format(i, min_norm_loss.item()))
        if best_checkpoint == {}:
            best_checkpoint = None
            min_loss = np.inf
        if return_normalized_loss:
            min_loss=min_norm_loss
        return best_checkpoint, min_loss

    def save_model(self, prep_model: Any, state: dict, optimizer_state: dict, name: Union[str, None] = None):
        """
        Saved model in a cache (uses for 'NN' and 'autograd' methods).
        Args:
            prep_model: model to save.
            state: a dict holding current model state (i.e., dictionary that maps each layer to its parameter tensor).
            optimizer_state: a dict holding current optimization state (i.e., values, hyperparameters).
            cache_dir: directory where saved cache in.
            name: name for a model.
        """
        if name == None:
            name = str(datetime.datetime.now().timestamp())
        if not(os.path.isdir(self.cache_dir)):
            os.mkdir(self.cache_dir)

        try:
            torch.save({'model': prep_model.to('cpu'), 'model_state_dict': state,
                        'optimizer_state_dict': optimizer_state}, self.cache_dir+'\\' + name + '.tar')
            print('model is saved in cache')
        except RuntimeError:
            torch.save({'model': prep_model.to('cpu'), 'model_state_dict': state,
                        'optimizer_state_dict': optimizer_state}, self.cache_dir+'\\' + name + '.tar', _use_new_zipfile_serialization=False) #cyrrilic in path
            print('model is saved in cache')
        except:
            print('Cannot save model in cache')


    def save_model_mat(self, name: None = None, cache_model: None = None,
                       cache_verbose: bool=False):
        """
        Saved model in a cache (uses for 'mat' method).

        Args:
            cache_dir: a directory where saved cache in.
            name: name for a model
            cache_model: model to save
        """

        NN_grid, cache_model = self.grid_model_mat(cache_model)
        optimizer = torch.optim.Adam(cache_model.parameters(), lr=0.001)
        model_res = self.model.reshape(-1, self.model.shape[0])

        def closure():
            optimizer.zero_grad()
            loss = torch.mean((cache_model(NN_grid) - model_res) ** 2)
            loss.backward()
            return loss

        loss = np.inf
        t = 0
        while loss > 1e-5 and t < 1e5:
            loss = optimizer.step(closure)
            t += 1
            if cache_verbose:
                print('Interpolate from trained model t={}, loss={}'.format(
                    t, loss))

        self.save_model(cache_model, cache_model.state_dict(),
                        optimizer.state_dict(), name=name)

    def scheme_interp(self, trained_model: Any, cache_verbose: bool = False) -> Tuple[Any, dict]:
        """
        Smth

        Args:
            trained_model: smth
            cache_verbose: detailed info about models in cache.
        Returns:
            * **model**  -- NN or mat.\n
            * **optimizer_state** -- dict.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        loss = torch.mean(torch.square(
            trained_model(self.grid) - self.model(self.grid)))

        def closure():
            optimizer.zero_grad()
            loss = torch.mean((
                trained_model(self.grid) - self.model(self.grid)) ** 2)
            loss.backward()
            return loss

        t = 0
        while loss > 1e-5 and t < 1e5:
            optimizer.step(closure)
            loss = torch.mean(torch.square(
                trained_model(self.grid) - self.model(self.grid)))
            t += 1
            if cache_verbose:
                print('Interpolate from trained model t={}, loss={}'.format(
                    t, loss))

        return self.model, optimizer.state_dict()

    def cache_retrain(self, cache_checkpoint, cache_verbose: bool = False) -> Union[
        Tuple[Any, None], Tuple[Any, Union[dict, Any]]]:
        """
        Smth
        Args:
            cache_checkpoint: smth
            cache_verbose: detailed info about models in cache.
        Returns:
            * **model** -- model.\n
            * **optimizer_state** -- smth
        """

        # do nothing if cache is empty
        if cache_checkpoint == None:
            optimizer_state = None
            return self.model, optimizer_state
        # if models have the same structure use the cache model state,
        # and the cache model has ordinary structure
        if str(cache_checkpoint['model']) == str(self.model) and \
                 isinstance(self.model, torch.nn.Sequential) and \
                 isinstance(self.model[0], torch.nn.Linear):
            self.model = cache_checkpoint['model']
            self.model.load_state_dict(cache_checkpoint['model_state_dict'])
            self.model.train()
            optimizer_state = cache_checkpoint['optimizer_state_dict']
            if cache_verbose:
                print('Using model from cache')
        # else retrain the input model using the cache model
        else:
            optimizer_state = None
            model_state = None
            cache_model = cache_checkpoint['model']
            cache_model.load_state_dict(cache_checkpoint['model_state_dict'])
            cache_model.eval()
            self.model, optimizer_state = self.scheme_interp(
                cache_model, cache_verbose=cache_verbose)
        return self.model, optimizer_state

    def cache_nn(self, nmodels: Union[int, None], lambda_operator: float, lambda_bound: float,
              cache_verbose: bool,model_randomize_parameter: Union[float, None],
              cache_model: torch.nn.Sequential, return_normalized_loss: bool = False):
        """
       Restores the model from the cache and uses it for retraining.
       Args:
           cache_dir: a directory where saved cache in.
           nmodels: smth
           lambda_bound: an arbitrary chosen constant, influence only convergence speed.
           cache_verbose: more detailed info about models in cache.
           model_randomize_parameter:  Creates a random model parameters (weights, biases) multiplied with a given
                                       randomize parameter.
           cache_model: cached model
       Returns:
           * **model** -- NN.\n
           * **min_loss** -- min loss as is.
       """
        r = create_random_fn(model_randomize_parameter)
        cache_checkpoint, min_loss = self.cache_lookup(nmodels=nmodels,
                                                       cache_verbose=cache_verbose,
                                                       lambda_operator= lambda_operator,
                                                       lambda_bound=lambda_bound, 
                                                       return_normalized_loss = return_normalized_loss)
        
        self.model, optimizer_state = self.cache_retrain(cache_checkpoint,
                                                         cache_verbose=cache_verbose)

        self.model.apply(r)

        return self.model, min_loss

    def cache_mat(self, nmodels: Union[int, None],lambda_operator: float, lambda_bound: float,
              cache_verbose: bool,model_randomize_parameter: Union[float, None],
              cache_model: torch.nn.Sequential, return_normalized_loss: bool = False):
        """
       Restores the model from the cache and uses it for retraining.
       Args:
           cache_dir: a directory where saved cache in.
           nmodels: smth
           lambda_bound: an arbitrary chosen constant, influence only convergence speed.
           cache_verbose: more detailed info about models in cache.
           model_randomize_parameter:  Creates a random model parameters (weights, biases) multiplied with a given
                                       randomize parameter.
           cache_model: cached model
       Returns:
           * **model** -- mat.\n
           * **min_loss** -- min loss as is.
       """

        NN_grid, cache_model = self.grid_model_mat(cache_model)
        operator = deepcopy(self.equal_cls.operator)
        bconds = deepcopy(self.equal_cls.bconds)
        operator = self.mat_op_coeff(operator)
        r = create_random_fn(model_randomize_parameter)
        eq = Equation(NN_grid, operator, bconds).set_strategy('autograd')
        model_cls = Model_prepare(NN_grid, eq, cache_model, 'autograd', self.weak_form)
        cache_checkpoint, min_loss = model_cls.cache_lookup(
            nmodels=nmodels,
            cache_verbose=cache_verbose,
            lambda_bound=lambda_bound,
            lambda_operator=lambda_operator,
            return_normalized_loss=return_normalized_loss)
        if cache_checkpoint is not None:
            prepared_model, optimizer_state = model_cls.cache_retrain(
                                                        cache_checkpoint,
                                                        cache_verbose=cache_verbose)

            prepared_model.apply(r)
        
            self.model = prepared_model(NN_grid).reshape(
                            self.model.shape).detach()

        min_loss, _ = Solution(self.grid, self.equal_cls,
                                      self.model, self.mode, self.weak_form,
                                      lambda_operator, lambda_bound, tol=0,
                                      derivative_points=2).evaluate()

        return self.model, min_loss

    def cache(self, nmodels: Union[int, None],lambda_operator, lambda_bound: float,
              cache_verbose: bool,model_randomize_parameter: Union[float, None],
              cache_model: torch.nn.Sequential, 
              return_normalized_loss: bool = False):
        """
        Restores the model from the cache and uses it for retraining.
        Args:
            cache_dir: a directory where saved cache in.
            nmodels: number cached models.
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
            cache_verbose: more detailed info about models in cache.
            model_randomize_parameter:  Creates a random model parameters (weights, biases) multiplied with a given
                                        randomize parameter.
            cache_model: cached model

        Returns:
            cache.cache_nn or cache.cache_mat
        """

        if self.mode != 'mat':
            return self.cache_nn(nmodels,lambda_operator, lambda_bound,
                                 cache_verbose, model_randomize_parameter,
                                 cache_model,return_normalized_loss=return_normalized_loss)
        elif self.mode == 'mat':
            return self.cache_mat(nmodels, lambda_operator, lambda_bound,
                                  cache_verbose, model_randomize_parameter,
                                  cache_model,return_normalized_loss=return_normalized_loss)

