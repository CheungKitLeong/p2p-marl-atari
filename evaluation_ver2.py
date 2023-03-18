import torch
from agent import Agent
from wrappers import make_pong
import numpy as np
import matplotlib.pyplot as plt
import csv

agent1_dir = 'models/2023-03-15_10_59_first_0/'
agent2_dir = 'models/2023-03-15_10_59_second_0/'

DQN_HYPERPARAMS = {
    'eps_start': 1,
    'eps_end': 0.1,
    'eps_decay': 10 ** 5,
    'buffer_size': 3000,
    'buffer_minimum': 1001,
    'learning_rate': 5e-5,
    'gamma': 0.9,
    'n_iter_update_nn': 1000,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("mps")

def evaluation(dir1, dir2, interval=20, max_epoch=600, eval_num=50, MAX_STEP=3000):
    env = make_pong()

    def fixed_opponent(fixed_player_epoch=400):
        win_rates = []
        penalties = []
        marks = []
        mean_rewards = []
        current_player = 0
        fixed_agent_path = agent1_dir + 'epoch_{}.pt'.format(fixed_player_epoch)
        fixed_agent = Agent(env, DQN_HYPERPARAMS, device, 'first_0', fixed_agent_path)

        while current_player <= max_epoch:
            agents = [fixed_agent, Agent(env, DQN_HYPERPARAMS, device, 'second_0',
                                         agent2_dir + 'epoch_{}.pt'.format(current_player))]
            mark = [0, 0]
            penalty = 0
            total_reward = 0
            for num in range(eval_num):
                print('fixed ,current_player:', current_player, 'eval_num:', num)
                env.reset()
                old_obs = [None] * 2

                for i in range(MAX_STEP):
                    done = False
                    trunc = False
                    for n in range(2):
                        new_obs, reward, done, trunc, info = env.last()
                        if n == 1:
                            total_reward += reward
                        if reward > 0:
                            mark[n] += reward

                        if (old_obs[n] is not None) and (new_obs == old_obs[n]).all() and (reward != 0):
                            action = 1
                            if n == 1:
                                penalty += 1
                        else:
                            action = agents[n].select_greedy_action(new_obs)
                        env.step(action)

                    if done or trunc:
                        break

            win_rates.append(float(mark[1]) / float(sum(mark)))
            marks.append(mark[1])
            penalties.append(penalty)
            mean_rewards.append(float(total_reward) / eval_num)

            print('fixed ,current_player:%d, win_rates:%f, mean_rewards:%f' % (current_player, float(mark[1]) / float(sum(mark)), float(total_reward) / eval_num))
            current_player += interval

        return win_rates, marks, penalties, mean_rewards

    print('Started fixed evaluation...')
    fixed = fixed_opponent()
    return fixed


def process(agent1_dir, agent2_dir):
    data = evaluation(agent1_dir, agent2_dir, interval=60, eval_num=10)

    with open('fixed_ver2.csv', 'w') as f1:
        writer = csv.writer(f1)
        writer.writerow(["win_rates", "marks", "ff", "mean_r"])
        for i in range(len(data[0])):
            writer.writerow([data[0][i], data[1][i], data[2][i], data[3][i]])

    plt.subplot(2, 2, 1)
    plt.plot([i * 20 for i in range(len(data[0]))], np.array(data[0]))
    plt.xlabel('model')
    plt.ylabel('win_rate')
    # plt.title('evaluating with fixed opponent')

    plt.subplot(2, 2, 2)
    plt.plot([i * 20 for i in range(len(data[1]))], np.array(data[1]))
    plt.xlabel('model')
    plt.ylabel('marks')
    # plt.title('evaluating with fixed opponent')

    plt.subplot(2, 2, 3)
    plt.plot([i * 20 for i in range(len(data[2]))], np.array(data[2]))
    plt.xlabel('model')
    plt.ylabel('penalties')
    p  # lt.title('evaluating with fixed opponent')

    plt.subplot(2, 2, 4)
    plt.plot([i * 20 for i in range(len(data[3]))], np.array(data[3]))
    plt.xlabel('model')
    plt.ylabel('mean_rewards')
    # plt.title('evaluating with fixed opponent')
    plt.savefig('result.png')


process(agent1_dir, agent2_dir)
