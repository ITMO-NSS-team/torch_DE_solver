# Separate file for RL algorithms (e.g., rl_algorithms.py)

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import math

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(6 * 6 * 32, 128)
        self.relu3 = nn.ReLU()
        self.fc2_optim = nn.Linear(128, 8)
        self.fc2_loss = nn.Linear(128, 5)
        self.fc2_epochs = nn.Linear(128, 5)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 6 * 6 * 32)
        x = self.relu3(self.fc1(x))
        x_optim = self.fc2_optim(x)
        x_loss = self.fc2_loss(x)
        x_epochs = self.fc2_epochs(x)
        return (x_optim, x_loss, x_epochs)



class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 memory_size=10000, batch_size=64):
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.steps_done = 0

        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def get_random_action(self):
        x_optim = random.randint(0, 7)
        x_loss = random.randint(0, 4)
        x_epochs = random.randint(0, 4)
        return (x_optim, x_loss, x_epochs)
    
    def optim_(self, buff_test):
        
        state, next_state, action_raw, reward =  zip(*buff_test)
        state = torch.stack(state, dim=0).reshape(-1,1,26,26)
        next_state = torch.stack(next_state, dim=0).reshape(-1,1,26,26)
        reward = torch.FloatTensor(reward)
        Q = self.model
        right_Q = Q(state)
        Q_ = self.target_model
        left_Q = Q_(next_state)

        optim_action, loss_action, epochs_action  = zip(*action_raw)
        right_Q_optim, right_Q_loss, right_Q_epochs = right_Q
        get_reward_by_action = lambda q_values, actions: q_values[torch.arange(len(actions)), actions]
        optim_action_Q = get_reward_by_action(right_Q_optim, optim_action)
        loss_action_Q = get_reward_by_action(right_Q_loss, loss_action)
        epochs_action_Q = get_reward_by_action(right_Q_epochs, epochs_action)
        left_Q_optim, left_Q_loss, left_Q_epochs = left_Q
        optim_max_Q = torch.max(left_Q_optim, dim=1).values
        loss_max_Q = torch.max(left_Q_loss, dim=1).values
        epochs_max_Q = torch.max(left_Q_epochs, dim=1).values
        reward_ = 1/reward
        dqn_optim = reward_ + GAMMA*optim_max_Q - optim_action_Q
        dqn_loss = reward_ + GAMMA*loss_max_Q - loss_action_Q
        dqn_epochs = reward_ + GAMMA*epochs_max_Q - epochs_action_Q
        loss_dqn = dqn_optim**2 + dqn_loss**2 + dqn_epochs**2
        loss = torch.mean(loss_dqn)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

    # Action function stub
    def select_action(self, state):

        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = state.reshape((1,26,26))
                x_optim, x_loss, x_epochs = self.model(state)
                x_optim, x_loss, x_epochs = torch.argmax(x_optim), torch.argmax(x_loss), torch.argmax(x_epochs)
                return (x_optim, x_loss, x_epochs)
        else:
            return self.get_random_action()
        # if random.random() < self.epsilon:
        #     return random.randint(0, self.action_dim - 1)  # choose optimizer and epochs
        # state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # with torch.no_grad():
        #     return torch.argmax(self.model(state_tensor)).item()

    def push_memory(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))

    def optimize_model(self):
        """Обучение сети на данных из буфера памяти."""
        if len(self.memory) < self.batch_size:
            return

        # transitions = self.memory.sample(self.batch_size)
        # batch = Transition(*zip(*transitions))

        # state_batch = torch.cat(batch.state)
        # action_batch = torch.cat(batch.action)
        # reward_batch = torch.cat(batch.reward)
        # next_state_batch = torch.cat(batch.next_state)

        # q_values = self.net(state_batch).gather(1, action_batch)

        # with torch.no_grad():
        #     next_q_values = self.net(next_state_batch).max(1)[0]
        #     expected_q_values = reward_batch + self.gamma * next_q_values

        # loss = self.loss_fn(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

