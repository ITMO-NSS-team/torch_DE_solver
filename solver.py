
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import datetime
from cache import Model_prepare
from device import check_device, device_type
from metrics import Solution
from cache import create_random_fn

def grid_format_prepare(coord_list, mode='NN'):
    device = device_type()
    if type(coord_list) == torch.Tensor:
        print('Grid is a tensor, assuming old format, no action performed')
        return check_device(coord_list)
    elif mode == 'NN' or mode == 'autograd':
        if len(coord_list) == 1:
            coord_list = torch.tensor(coord_list).float().to(device)
            grid = coord_list.reshape(-1, 1)
        else:
            coord_list_tensor = []
            for item in coord_list:
                if isinstance(item, (np.ndarray)):
                    coord_list_tensor.append(torch.from_numpy(item).to(device))
                else:
                    coord_list_tensor.append(item.to(device))
            grid=torch.cartesian_prod(*coord_list_tensor)
    elif mode=='mat':
        grid = np.meshgrid(*coord_list)
        grid = torch.tensor(np.array(grid)).to(device)
    return grid


class Solver():
    def __init__(self, grid, equal_cls, model, mode, weak_form=None):
        self.grid = check_device(grid)
        self.equal_cls = equal_cls
        self.model = model
        self.mode = mode
        self.weak_form = weak_form

    def optimizer_choice(self, optimizer, learning_rate):
        if optimizer=='Adam':
            torch_optim = torch.optim.Adam
        elif optimizer=='SGD':
            torch_optim = torch.optim.SGD
        elif optimizer=='LBFGS':
            torch_optim = torch.optim.LBFGS
        else:
            print('Wrong optimizer chosen, optimization was not performed')
            return self.model

        if self.mode =='NN' or self.mode == 'autograd':
            optimizer = torch_optim(self.model.parameters(), lr=learning_rate)
        elif self.mode =='mat':
            optimizer = torch_optim([self.model.requires_grad_()], lr=learning_rate)
       
        return optimizer

    def print_nn(self, title):
        nvars_model = self.model[-1].out_features
        nparams = self.grid.shape[1]
        fig = plt.figure()
        for i in range(nvars_model):
            if nparams == 1:
                ax1 = fig.add_subplot(1,nvars_model,i+1)
                if title != None:
                    ax1.set_title(title+' variable {}'.format(i))
                ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                            self.model(self.grid)[:,i].detach().cpu().numpy())
            else:
                ax1 = fig.add_subplot(1, nvars_model, i+1, projection='3d')
                if title != None:
                    ax1.set_title(title+' variable {}'.format(i))
                ax1.plot_trisurf(self.grid[:, 0].detach().cpu().numpy(), 
                            self.grid[:, 1].detach().cpu().numpy(),
                            self.model(self.grid)[:,i].detach().cpu().numpy(),
                            cmap=cm.jet, linewidth=0.2, alpha=1)
                ax1.set_xlabel("x1")
                ax1.set_ylabel("x2")

    def print_mat(self, title):
        nparams = self.grid.shape[0]
        if nparams == 1:
            fig = plt.figure()
            plt.scatter(self.grid.cpu().reshape(-1),
                        self.model.detach().cpu().numpy().reshape(-1))
        elif nparams == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if title!=None:
                ax.set_title(title)
            ax.plot_trisurf(self.grid[0].cpu().reshape(-1),
                            self.grid[1].cpu().reshape(-1),
                            self.model.reshape(-1).detach().cpu().numpy(),
                            cmap=cm.jet, linewidth=0.2, alpha=1)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")

    def dir_path(self, save_dir):
        if save_dir == None:
            try:
                img_dir = os.path.join(os.path.dirname( __file__ ), 'img')
            except:
                current_dir = globals()['_dh'][0]
                img_dir = os.path.join(os.path.dirname(current_dir), 'img')
            
            if not (os.path.isdir(img_dir)):
                os.mkdir(img_dir)
            directory = os.path.abspath(os.path.join(img_dir, 
                            str(datetime.datetime.now().timestamp())+'.png'))
        else:
            if not (os.path.isdir(save_dir)):
                os.mkdir(save_dir)
            directory = os.path.join(save_dir,
                            str(datetime.datetime.now().timestamp())+'.png')
        return directory

    def solution_print(self, title=None, solution_print=False,
                       solution_save=False, save_dir=None):

        directory = self.dir_path(save_dir)

        if self.mode == 'mat':
            self.print_mat(title)
        else:
            self.print_nn(title)
        if solution_save:
            plt.savefig(directory)
        if solution_print:
            plt.show()
        plt.close()

    def solve(self, lambda_bound=10, verbose=False, learning_rate=1e-4,
              eps=1e-5, tmin=1000, tmax=1e5, nmodels=None, name=None,
              abs_loss=None, use_cache=True, cache_dir='../cache/',
              cache_verbose=False, save_always=False, print_every=100,
              cache_model=None, patience=5, loss_oscillation_window=100,
              no_improvement_patience=1000, model_randomize_parameter=0,
              optimizer_mode='Adam', step_plot_print=False,
              step_plot_save=False,image_save_dir=None):
        

        Cache_class = Model_prepare(self.grid, self.equal_cls,
                                                        self.model, self.mode)

        # prepare input data to uniform format 
        r = create_random_fn(model_randomize_parameter)
        #  use cache if needed
        if use_cache:
            self.model, min_loss = Cache_class.cache(cache_dir, nmodels,
                                                     lambda_bound,
                                                     cache_verbose,
                                                     model_randomize_parameter,
                                                     cache_model,
                                                     weak_form=self.weak_form,)
            
            Solution_class = Solution(self.grid, self.equal_cls,
                                      self.model, self.mode)
        else:
            Solution_class = Solution(self.grid, self.equal_cls,
                                      self.model, self.mode)
            min_loss = Solution_class.loss_evaluation(lambda_bound=lambda_bound,
                                                      weak_form=self.weak_form)
        
        optimizer = self.optimizer_choice(optimizer_mode, learning_rate)
    
        # standard NN stuff
        if verbose:
            print('[{}] initial (min) loss is {}'.format(
                datetime.datetime.now(), min_loss))
    
        t = 0
    
        last_loss = np.zeros(loss_oscillation_window) + float(min_loss)
        line = np.polyfit(range(loss_oscillation_window), last_loss, 1)

        def closure():
            nonlocal cur_loss
            optimizer.zero_grad()
            loss = Solution_class.loss_evaluation(
                lambda_bound = lambda_bound, weak_form=self.weak_form)
            loss.backward()
            cur_loss = loss.item()
            return loss

        stop_dings = 0
        t_imp_start = 0
        # to stop train proceduce we fit the line in the loss data
        # if line is flat enough 5 times, we stop the procedure
        cur_loss = min_loss
        while stop_dings <= patience:
            optimizer.step(closure)

            if cur_loss != cur_loss:
                print(f'Loss is equal to NaN, something went wrong (LBFGS+high'
                 f'leraning rate and pytorch<1.12 could be the problem)')
                break

            last_loss[(t-1)%loss_oscillation_window] = cur_loss

        
            if cur_loss < min_loss:
                min_loss = cur_loss
                t_imp_start = t

            if verbose:
                info_string='Step = {} loss = {:.6f} normalized loss line= {:.6f}x+{:.6f}. There was {} stop dings already.'.format(
                                                                        t, cur_loss, line[0]/cur_loss, line[1]/cur_loss, stop_dings+1)

            if t%loss_oscillation_window == 0:
                line = np.polyfit(range(loss_oscillation_window), last_loss, 1)
                if abs(line[0]/cur_loss) < eps and t > 0:
                    stop_dings += 1
                    if self.mode =='NN' or self.mode =='autograd':
                        self.model.apply(r)
                    if verbose:
                        print('[{}] Oscillation near the same loss'.format(
                            datetime.datetime.now()))
                        print(info_string)
                        if step_plot_print or step_plot_save:
                            self.solution_print(title='Iteration = ' + str(t),
                                                solution_print=step_plot_print,
                                                solution_save=step_plot_save,
                                                save_dir=image_save_dir)

            if (t-t_imp_start) == no_improvement_patience:
                if verbose:
                    print('[{}] No improvement in {} steps'.format(
                        datetime.datetime.now(),no_improvement_patience))
                    print(info_string)
                    if step_plot_print or step_plot_save:
                        self.solution_print(title='Iteration = ' + str(t),
                                                solution_print=step_plot_print,
                                                solution_save=step_plot_save,
                                                save_dir=image_save_dir)
                t_imp_start = t
                stop_dings += 1
                if self.mode == 'NN' or self.mode == 'autograd':
                        self.model.apply(r)


            if abs_loss != None and cur_loss < abs_loss:
                if verbose:
                    print('[{}] Absolute value of loss is lower than threshold'.format(datetime.datetime.now()))
                    print(info_string)
                    if step_plot_print or step_plot_save:
                        self.solution_print(title='Iteration = ' + str(t),
                                                solution_print=step_plot_print,
                                                solution_save=step_plot_save,
                                                save_dir=image_save_dir)
                stop_dings += 1


            if print_every != None and (t % print_every == 0) and verbose:
                print('[{}] Print every {} step'.format(
                    datetime.datetime.now(),print_every))
                print(info_string)
                if step_plot_print or step_plot_save:
                    self.solution_print(title='Iteration = ' + str(t),
                                                solution_print=step_plot_print,
                                                solution_save=step_plot_save,
                                                save_dir=image_save_dir)
            t += 1
            if t > tmax:
                break
        if save_always:
            if self.mode == 'mat':
                Cache_class.save_model_mat(cache_dir=cache_dir, name=name)
            else:
                Cache_class.save_model(self.model, self.model.state_dict(),
                                optimizer.state_dict(), cache_dir=cache_dir,
                                name=name)
        return self.model






