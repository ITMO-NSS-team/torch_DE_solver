# Separate file for RL algorithms (e.g., rl_algorithms.py)

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import math
from copy import copy
import wandb
import matplotlib.pyplot as plt
from collections import defaultdict


GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'action', 'reward', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque()

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        current_sample = random.sample(self.memory, batch_size)
        current_sample_tuples = [tuple(t) for t in current_sample]
        return current_sample_tuples

    def __len__(self):
        return len(self.memory)
    
class DQN_optim(nn.Module):
    def __init__(self, optim_n):
        super(DQN_optim, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(6 * 6 * 32, 256)  # n_observation instead 6 * 6
        self.relu3 = nn.ReLU()
        self.fc_optim_class = nn.Linear(256, optim_n)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 6 * 6 * 32)
        x = self.relu3(self.fc1(x))
        x_optim = self.softmax(self.fc_optim_class(x))
        return x_optim, x
    
class DQN_params(nn.Module):
    def __init__(self, optimizer_dict):
        super(DQN_params, self).__init__()
        self.optimizer_dict = optimizer_dict
        layers_ar = []
        self.fc_liner = lambda param_var: nn.Linear(256, len(param_var))
        self.fc_param_by_opt = defaultdict(defaultdict)
        for opt_name in self.optimizer_dict.keys():
            for param_name in self.optimizer_dict[opt_name].keys():
                param_var = self.optimizer_dict[opt_name][param_name]
                # self.fc_param_by_opt[opt_name][param_name] = nn.Linear(128, len(param_var))
                linear_layer = self.fc_liner(param_var)
                self.fc_param_by_opt[opt_name][param_name] = linear_layer
                layers_ar.append(linear_layer)
        self.linears = nn.ModuleList(layers_ar)
            
        self.softmax = nn.Softmax()

    def forward(self, x, optim_name_ar):
        x_params_ar = []
        for i, optim_name in enumerate(optim_name_ar):
            x_params = {}
            for param in self.fc_param_by_opt[optim_name].keys():
                param_liner = self.fc_param_by_opt[optim_name][param]
                x_params[param] = self.softmax(param_liner(x[i]))
            x_params_ar.append(x_params)
        return x_params_ar

# class DQN(nn.Module):
#     def __init__(self, n_observation, optimizer_dict):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.n_observation = n_observation
#         self.optimizer_dict = optimizer_dict

#         self.fc1 = nn.Linear(6 * 6 * 32, 128)  # n_observation instead 6 * 6
#         self.relu3 = nn.ReLU()
#         self.fc_optim_class = nn.Linear(128, len(self.optimizer_dict.keys()))
#         self.opt2class = {}
#         self.param2class = {}
#         self.fc_param_by_opt = defaultdict(defaultdict)
#         opt_i = 0
#         for opt_name in self.optimizer_dict.keys():
#             param_i = 0
#             self.opt2class[opt_i] = opt_name
#             for param_name in self.optimizer_dict[opt_name].keys():
#                 param_var = self.optimizer_dict[opt_name][param_name]
#                 if param_name not in self.param2class:
#                     self.param2class[param_i] = param_name
#                 self.fc_param_by_opt[opt_name][param_name] = nn.Linear(128, len(param_var))
#                 param_i += 1
#             opt_i += 1

#         self.softmax = nn.Softmax()

#     def forward(self, x):
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         x = x.view(-1, 6 * 6 * 32)
#         x = self.relu3(self.fc1(x))
#         x_optim = self.softmax(self.fc_optim_class(x))
#         optim_name_ar = [self.opt2class[int(el)] for el in torch.argmax(x_optim, dim=1)]
#         x_params_ar = []
#         for i_optim, optim_name in enumerate(optim_name_ar):
#             x_params = {}
#             for param in self.fc_param_by_opt[optim_name].keys():
#                 param_liner = self.fc_param_by_opt[optim_name][param]
#                 x_params[param] = self.softmax(param_liner(x[i_optim]))
#             x_params_ar.append(x_params)
#         return x_optim, x_params_ar


class DQNAgent:
    def __init__(self, n_observation=None, n_action=None, optimizer_dict=None, lr=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000, batch_size=128, device='cpu'):
        self.n_observation = n_observation
        self.n_action = n_action
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(memory_size)
        self.replay_buffer_copy = None
        self.steps_done = 0
        self.opt_count = 0
        self.opt_count_out = 0
        self.optimizer_dict = optimizer_dict
        self.i2opt = {v: k for v, k in enumerate(optimizer_dict.keys())}
        uniq_params = list(set([x for xs in optimizer_dict.values() for x in xs]))
        self.i2params = {k: v for v, k in enumerate(uniq_params)}
        self.huberloss = nn.HuberLoss()

        self.device = device

        self.model_optim = DQN_optim(len(self.i2opt)).to(self.device)
        self.model_params = DQN_params(self.optimizer_dict).to(self.device)

        self.reinit_target()

        self.optimizer_opt = optim.Adam(self.model_optim.parameters(), lr=lr)
        self.optimizer_params = optim.Adam(self.model_params.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def reinit_target(self):
        self.target_model_optim = DQN_optim(len(self.i2opt)).to(self.device)
        self.target_model_params = DQN_params(self.optimizer_dict).to(self.device)
        for param in self.target_model_optim.parameters():
            param.requires_grad = False
        for param in self.target_model_params.parameters():
            param.requires_grad = False
        self.target_model_optim.load_state_dict(self.model_optim.state_dict())
        self.target_model_optim.eval()
        self.target_model_params.load_state_dict(self.model_params.state_dict())
        self.target_model_params.eval()

    def detach_transition(self, transition):
        def detach_item(item):
            if isinstance(item, torch.Tensor):
                return item.detach().clone()
            elif isinstance(item, tuple):
                return tuple(detach_item(subitem) for subitem in item)
            elif isinstance(item, dict):
                return {k: detach_item(v) for k, v in item.items()}
            else:
                return item

        return Transition(
            state=detach_item(transition.state),
            next_state=detach_item(transition.next_state),
            action=detach_item(transition.action),
            reward=detach_item(transition.reward),
            done=detach_item(transition.done)
        )

    def deepcopy_replay_buffer_without_graph(self, buffer):
        clean_buffer = ReplayBuffer(capacity=len(buffer.memory))
        for transition in buffer.memory:
            clean_buffer.push(*self.detach_transition(transition))
        return clean_buffer

    def optim_(self):
        loss_arr_optim_class = []
        loss_arr_param = []
        # self.replay_buffer_copy = deepcopy(self.replay_buffer)
        self.replay_buffer_copy = self.deepcopy_replay_buffer_without_graph(self.replay_buffer)

        while len(self.replay_buffer_copy.memory) >= self.batch_size:
            buff_test = self.replay_buffer_copy.sample(self.batch_size)

            transition_equal = lambda t1, t2: (
                        all(torch.equal(t1.state[k], t2[0][k]) for k in t1.state) and
                        all(torch.equal(t1.next_state[k], t2[1][k]) for k in t1.next_state) and
                        t1.action == t2[2] and
                        torch.equal(t1.reward, t2[3]) and
                        t1.done == t2[4]
                )

            self.replay_buffer_copy.memory = deque(
                filter(
                    lambda x: all(not transition_equal(x, y) for y in buff_test), self.replay_buffer_copy.memory
                )
            )

            state, next_state, action, reward, done = zip(*buff_test)

            # state = {'loss_total': tensor([...]), 'loss_oper': tensor([...]), 'loss_bnd': tensor([...])}
            state = [torch.stack((elem['loss_oper'], elem['loss_bnd']), dim=0) for elem in state]
            next_state = [torch.stack((elem['loss_oper'], elem['loss_bnd']), dim=0) for elem in next_state]
            state = torch.stack(state, dim=0).reshape(-1, 2, 26, 26).to(self.device)
            next_state = torch.stack(next_state, dim=0).reshape(-1, 2, 26, 26).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            done = torch.IntTensor(done).to(self.device)

            target_optim, liner_out_target = self.target_model_optim(next_state)
            model_optim, liner_out_model = self.model_optim(state)
            
            targets = lambda reward, done, target_res: \
                    reward + (1 - done) * GAMMA * torch.max(target_res, dim=1).values
            q_values = lambda model_res, action_: \
                    model_res[torch.arange(self.batch_size), action_]
            
            q_values_optim = targets(reward, done, target_optim)
            targets_optim = q_values(model_optim, list(zip(*action))[0])
            loss = self.huberloss(input=q_values_optim, target=targets_optim)
            loss_arr_optim_class.append(loss)

            self.optimizer_opt.zero_grad()
            loss.backward()
            self.optimizer_opt.step()

            action_clases = [self.i2opt[el] for el in list(zip(*action))[0]]

            liner_out_target = liner_out_target.detach()
            liner_out_target.requires_grad = True
            liner_out_model = liner_out_model.detach()
            liner_out_model.requires_grad = True

            target_params = self.target_model_params(liner_out_target, action_clases)
            model_params = self.target_model_params(liner_out_model, action_clases)
            q_values_optim_ar = []
            targets_optim_ar = []
            for i, optim_name in enumerate(action_clases):
                for param_name in self.optimizer_dict[optim_name].keys():
                    q_values_optim_ar.append(targets(reward[i], done[i], target_params[i][param_name].reshape(1,-1)))
                    action_ = action[i][1][param_name]
                    q_values_dist = model_params[i][param_name]
                    targets_optim_ar.append(q_values_dist[action_])
            # for key in 
            # loss = (q_values - targets) ** 2
            q_values_optim = torch.stack(q_values_optim_ar)
            targets_optim = torch.stack(targets_optim_ar)

            loss = self.huberloss(input=q_values_optim, target=targets_optim)
            loss_arr_param.append(loss)
            # loss = loss.type(torch.DoubleTensor)
            # loss = torch.mean(loss)

            self.optimizer_params.zero_grad()
            loss.backward()
            self.optimizer_params.step()
            print("\nRL optimization is complete!\n")

        mean_batch_loss_optim_class = 0
        for el in loss_arr_optim_class:
            mean_batch_loss_optim_class += el
        mean_batch_loss_optim_class = mean_batch_loss_optim_class / len(loss_arr_optim_class)
        if loss_arr_param != []:
            mean_batch_loss_param = 0
            for el in loss_arr_param:
                mean_batch_loss_param += el
            mean_batch_loss_param = mean_batch_loss_param / len(loss_arr_param)
        wandb.log({"mean_batch_loss_optim_class": mean_batch_loss_optim_class,\
                "mean_batch_loss_param": mean_batch_loss_param, "steps_done": self.steps_done})

        # self.replay_buffer.memory = deque(filter(lambda x: x not in set(buff_test), self.replay_buffer.memory),
        #                                   maxlen=self.replay_buffer.memory.maxlen)

        self.opt_count += 1
        self.opt_count_out += 1
        if self.opt_count == 5:
            self.reinit_target()
            self.opt_count = 0
        return mean_batch_loss_optim_class
    
    def post_proc_model(self, optim_class, epochs_class, param_class):
        class_name = self.i2opt[optim_class]
        epochs = self.optimizer_dict[class_name]['epochs'][epochs_class]
        params = {}
        for param_name, param_val in param_class.items():
            params[param_name] = self.optimizer_dict[class_name][param_name][param_val]
        action_dict = {
            'type': class_name,
            'epochs': epochs,
            'params': params
        }
        return action_dict


    def get_random_action(self):
        optim_class = random.randint(0, len(self.i2opt) - 1)
        class_name = self.i2opt[optim_class]
        param_class = {}
        optim_class_dict = self.optimizer_dict[class_name]

        for key in optim_class_dict:
            if key == 'epochs': epochs_class = random.randint(0, len(optim_class_dict['epochs']) - 1)
            else:
                param_class[key] = random.randint(0, len(optim_class_dict[key]) - 1)

        return optim_class, epochs_class, param_class
    
    # Action function stub
    def select_action(self, state):
        with torch.no_grad():
            # state = state['loss_total'].to(self.device)
            state = torch.stack((state['loss_oper'], state['loss_bnd']), dim=0)
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * self.steps_done / EPS_DECAY)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    state = state.reshape((-1, 26, 26))
                    # x_optim, x_loss, x_epochs = self.model(state)
                    # x_optim, x_loss, x_epochs = torch.argmax(x_optim), torch.argmax(x_loss), torch.argmax(x_epochs)
                    x, liner_out = self.model_optim(state)
                    optim_class = int(torch.argmax(x))
                    optim_class_name = self.i2opt[optim_class]
                    param_class = {}
                    param_dict = self.model_params(liner_out, [optim_class_name])[0]
                    for key in param_dict:
                        if key == 'epochs': epochs_class = torch.argmax(param_dict[key])
                        else: param_class[key] = torch.argmax(param_dict[key])
            else:
                optim_class, epochs_class, param_class = self.get_random_action()
            action = self.post_proc_model(int(optim_class), epochs_class, param_class)
            return action, (int(optim_class), epochs_class, param_class)

    def push_memory(self, rl_params):
        # self.replay_buffer.memory += (rl_params,)
        self.replay_buffer.push(*rl_params)

    def render_Q_function(self):
        x = nn.Sigmoid()(torch.sum(self.model.fc_last.weight, dim=1))
        x = x.detach().numpy() 
        plt.figure(figsize=(10, 6))
        plt.bar([i for i in range(len(x))],x,align='center')
        plt.savefig(f'q_function_steps_{self.opt_count_out}.png')
