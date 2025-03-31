# Separate file for RL algorithms (e.g., rl_algorithms.py)

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import math

GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        current_sample = random.sample(self.memory, batch_size)
        current_sample_tuples = [tuple(t) for t in current_sample]
        self.memory = deque(filter(lambda x: x not in set(current_sample), self.memory), maxlen=self.memory.maxlen)
        return current_sample_tuples

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observation, n_action):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(n_observation * 32, 128)  # n_observation instead 6 * 6
        self.relu3 = nn.ReLU()
        self.fc2_optim = nn.Linear(128, n_action["type"])
        self.fc2_loss = nn.Linear(128, n_action["params"])
        self.fc2_epochs = nn.Linear(128, n_action["epochs"])
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 6 * 6 * 32)
        x = self.relu3(self.fc1(x))
        x_optim = self.fc2_optim(x)
        x_loss = self.fc2_loss(x)
        x_epochs = self.fc2_epochs(x)
        return x_optim, x_loss, x_epochs


class DQNAgent:
    def __init__(self, n_observation=None, n_action=None, lr=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000, batch_size=128, device='cpu'):
        self.n_observation = n_observation
        self.n_action = n_action
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(memory_size)
        self.steps_done = 0

        self.device = device

        self.model = DQN(self.n_observation, self.n_action).to(self.device)

        self.target_model = DQN(self.n_observation, self.n_action).to(self.device)
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def get_random_action(self):
        x_optim = random.randint(0, 6)
        x_loss = random.randint(0, 3)
        x_epochs = random.randint(0, 2)

        return x_optim, x_loss, x_epochs

    def optim_(self):
        while len(self.replay_buffer.memory) >= self.batch_size:
            buff_test = self.replay_buffer.sample(self.batch_size)

            state, next_state, action_raw, reward = zip(*buff_test)
            state = torch.stack(state, dim=0).reshape(-1, 1, 26, 26).to(self.device)
            next_state = torch.stack(next_state, dim=0).reshape(-1, 1, 26, 26).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)

            Q = self.model
            right_Q = Q(state)
            Q_ = self.target_model
            left_Q = Q_(next_state)

            optim_action, loss_action, epochs_action = zip(*action_raw)
            right_Q_optim, right_Q_loss, right_Q_epochs = right_Q
            get_reward_by_action = lambda q_values, actions: q_values[torch.arange(len(actions)), actions]
            optim_action_Q = get_reward_by_action(right_Q_optim, optim_action)
            loss_action_Q = get_reward_by_action(right_Q_loss, loss_action)
            epochs_action_Q = get_reward_by_action(right_Q_epochs, epochs_action)
            left_Q_optim, left_Q_loss, left_Q_epochs = left_Q
            optim_max_Q = torch.max(left_Q_optim, dim=1).values
            loss_max_Q = torch.max(left_Q_loss, dim=1).values
            epochs_max_Q = torch.max(left_Q_epochs, dim=1).values
            reward_ = reward
            dqn_optim = reward_ + GAMMA * optim_max_Q - optim_action_Q
            dqn_loss = reward_ + GAMMA * loss_max_Q - loss_action_Q
            dqn_epochs = reward_ + GAMMA * epochs_max_Q - epochs_action_Q
            loss_dqn = dqn_optim ** 2 + dqn_loss ** 2 + dqn_epochs ** 2
            loss = torch.mean(loss_dqn)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("\nRL optimization is complete!\n")

    # Action function stub
    def select_action(self, state):
        state = state.to(self.device)
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = state.reshape((1, 26, 26))
                x_optim, x_loss, x_epochs = self.model(state)
                x_optim, x_loss, x_epochs = torch.argmax(x_optim), torch.argmax(x_loss), torch.argmax(x_epochs)
                return (x_optim, x_loss, x_epochs)
        else:
            return self.get_random_action()

    def push_memory(self, rl_params):
        # self.replay_buffer.memory += (rl_params,)
        self.replay_buffer.push(*rl_params)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
