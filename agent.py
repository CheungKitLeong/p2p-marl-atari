import torch
import numpy as np
from neural_nets import DQN
import torch.optim as optim
import torch.nn as nn

class AgentControl:

    def __init__(self, env, device, lr, gamma, multi_step, double_dqn):
        self.env = env
        self.device = device
        self.gamma = gamma
        # We need to send both NNs to GPU hence '.to("cuda")
        self.moving_nn = DQN(input_shape = env.observation_space.shape, num_of_actions = env.action_space.n).to(device)
        self.target_nn = DQN(input_shape = env.observation_space.shape, num_of_actions = env.action_space.n).to(device)
        self.target_nn.load_state_dict(self.moving_nn.state_dict())

        self.optimizer = optim.AdamW(self.moving_nn.parameters(), lr=lr)
        self.loss = nn.MSELoss()