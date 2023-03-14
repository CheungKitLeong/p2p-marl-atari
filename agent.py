from agent_control import AgentControl
from replay_buffer import ReplayBuffer
from collections import namedtuple
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch import save, load
from pathlib import Path
import json


class Agent:
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'),
                            rename=False)  # 'rename' means not to overwrite invalid field

    def __init__(self, env, hyperparameters, device, name, load_path=None):
        if load_path is not None:
            self.agent_control = AgentControl(env, device, 0, 0, name)
            self.agent_control.moving_nn.load_state_dict(load(load_path, map_location=device))
            return

        self.eps_start = hyperparameters['eps_start']
        self.eps_end = hyperparameters['eps_end']
        self.eps_decay = hyperparameters['eps_decay']
        self.epsilon = hyperparameters['eps_start']
        self.n_iter_update_nn = hyperparameters['n_iter_update_nn']
        self.env = env
        self.name = name

        self.agent_control = AgentControl(env, device, hyperparameters['learning_rate'], hyperparameters['gamma'], name)
        self.replay_buffer = ReplayBuffer(hyperparameters['buffer_size'], hyperparameters['buffer_minimum'],
                                          hyperparameters['gamma'])
        self.path = datetime.now().strftime("%Y-%m-%d_%H_%M_") + name
        Path('models/' + self.path).mkdir(parents=True, exist_ok=True)
        with open('models/' + self.path + '/hparams.json', 'w') as file:
            json.dump(hyperparameters, file)
        self.summary_writer = SummaryWriter('logs/' + self.path)

        self.num_iterations = 0
        self.total_reward = 0
        self.num_games = 0
        self.total_loss = []
        self.ts_frame = 0
        self.ts = time.time()
        self.birth_time = time.time()
        self.rewards = []
        self.defend_frame = 0
        self.defend_frames = []

    def select_greedy_action(self, obs):
        # Give current state to the control who will pass it to NN which will
        # return all actions and the control will take max and return it here
        return self.agent_control.select_greedy_action(obs)

    def select_eps_greedy_action(self, obs):
        rand_num = np.random.rand()
        if self.epsilon > rand_num:
            # Select random action - explore
            return self.env.action_space(self.name).sample()
        else:
            # Select best action
            return self.select_greedy_action(obs)

    def add_to_buffer(self, obs, action, new_obs, reward, done):
        transition = self.Transition(state=obs, action=action, next_state=new_obs, reward=reward, done=done)
        if reward == 0:
            self.defend_frame += 1
        else:
            self.defend_frames.append(self.defend_frame)
            self.defend_frame = 0

        # print("%s: %d" % (self.name, self.defend_frame))
        self.replay_buffer.append(transition)
        self.num_iterations = self.num_iterations + 1
        if self.epsilon > self.eps_end:
            self.epsilon = self.eps_start - self.num_iterations / self.eps_decay
        self.total_reward = self.total_reward + reward

    def sample_and_improve(self, batch_size):
        # If buffer is big enough
        if len(self.replay_buffer.buffer) > self.replay_buffer.minimum:
            # Sample batch_size number of transitions from buffer B
            mini_batch = self.replay_buffer.sample(batch_size)
            # Calculate loss and improve NN
            loss = self.agent_control.improve(mini_batch)
            # So we can calculate mean of all loss during one game
            self.total_loss.append(loss)

        if (self.num_iterations % self.n_iter_update_nn) == 0:
            self.agent_control.update_target_nn()

    def reset_parameters(self):
        self.total_reward = 0
        self.num_games = self.num_games + 1
        self.total_loss = []
        self.defend_frame = 0
        self.defend_frames = []

    def print_info(self):
        self.rewards.append(self.total_reward)
        # print(self.num_iterations, self.ts_frame, time.time(), self.ts)
        fps = (self.num_iterations - self.ts_frame) / (time.time() - self.ts)
        print('%d %d %s: rew:%d fps:%d, eps:%.2f, loss:%.4f' % (
            self.num_iterations, self.num_games, self.name, self.total_reward, fps, self.epsilon,
            np.mean(self.total_loss)))
        self.ts_frame = self.num_iterations
        self.ts = time.time()

        if self.summary_writer is not None:
            self.summary_writer.add_scalar('reward', self.total_reward, self.num_games)
            self.summary_writer.add_scalar('mean_reward', np.mean(self.rewards[-40:]), self.num_games)
            self.summary_writer.add_scalar('10_mean_reward', np.mean(self.rewards[-10:]), self.num_games)
            self.summary_writer.add_scalar('epsilon', self.epsilon, self.num_games)
            self.summary_writer.add_scalar('loss', np.mean(self.total_loss), self.num_games)
            self.summary_writer.add_scalar('defend_frame', np.mean(self.defend_frames), self.num_games)

        # Save the model dict
        # Create folder to save models
        if self.num_games % 20 == 0:
            path = 'models/' + self.path + '/epoch_' + str(self.num_games) + '.pt'
            save(self.agent_control.moving_nn.state_dict(), path)
