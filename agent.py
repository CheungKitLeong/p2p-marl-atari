import torch
import numpy as np
from neural_nets import DQN
import torch.optim as optim
import torch.nn as nn


class Agent:
    def __init__(self, env, device, lr, gamma, eps):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.eps = eps
        # We need to send both NNs to GPU hence '.to("cuda")
        self.moving_nn = DQN(input_shape = env.observation_space.shape, num_of_actions = env.action_space.n).to(device)
        self.target_nn = DQN(input_shape = env.observation_space.shape, num_of_actions = env.action_space.n).to(device)
        self.target_nn.load_state_dict(self.moving_nn.state_dict())

        self.optimizer = optim.AdamW(self.moving_nn.parameters(), lr=lr)
        self.loss = nn.MSELoss()


    def select_eps_greedy_action(self, state):
        pass

    def select_greedy_action(self, state):
        pass

    def optimize_model(self, mini_batch):
        # Compute loss

        # Transform numpy array to Tensor and send it to GPU
        states_tensor = torch.as_tensor(states).to(self.device)
        next_states_tensor = torch.as_tensor(next_states).to(self.device)
        actions_tensor = torch.as_tensor(actions).to(self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
        done_tensor = torch.as_tensor(dones, dtype=torch.uint8).to(self.device)
        
        curr_state_action_value = self.moving_nn(states_tensor).gather(1,actions_tensor[:,None]).squeeze(-1)
        next_state_action_value = self.target_nn(next_states_tensor).max(1)[0]
        # We do differentiation for moving_nn (w or curr_state_action_value) and we dont do it for target_nn (w'),
        # so we dont have to remember operations for backprop. Good for huge amount of operations
        next_state_action_value = next_state_action_value.detach()

        q_target = rewards_tensor + self.gamma * next_state_action_value

        training_loss = self.loss(curr_state_action_value, q_target)

        self.optimizer.zero_grad()
        training_loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

        return loss.item()


    