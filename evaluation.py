import torch
from agent import Agent
from wrappers import make_pong
import numpy as np
import matplotlib.pyplot as plt

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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")

def evaluation(dir1, dir2, interval=20, max_epoch=20, eval_num=1, MAX_STEP=10000):
    env = make_pong(True)
    def fixed_opponent(fixed_player_epoch=300):
        fixed_r = []
        current_player = 0
        fixed_agent_path = agent1_dir + 'epoch_{}.pt'.format(fixed_player_epoch)
        fixed_agent = Agent(env, DQN_HYPERPARAMS, device, 'first_0', fixed_agent_path)

        while current_player <= max_epoch:
            agents = [fixed_agent, Agent(env, DQN_HYPERPARAMS, device, 'second_0', agent2_dir + 'epoch_{}.pt'.format(600))]
            r = []
            for num in range(eval_num):
                print('fixed ,current_player:', current_player, 'eval_num:', num)
                env.reset()
                # env.step(4)
                # env.step(4)

                for i in range(MAX_STEP):
                    for n in range(2):
                        obs, reward, done, trunc, info = env.last()
                        action = agents[n].select_greedy_action(obs)
                        env.step(action)
                        if(env.last()[1] > 0):
                            r.append(n)

                        # if reward > 0 or reward < 0 and not done and not trunc:
                        #     print(reward)
                        #     env.step(1)
                        #     env.step(1)

                    if done or trunc:
                        if trunc:
                            r.append(-1)
                        break
            print('fixed ,current_player:', current_player, 'mean:', np.mean(r))
            fixed_r.append(r)
            current_player += interval
        return fixed_r

    def adjacent_opponent():
        adjacent_r = []
        current_player = 20

        while current_player <= max_epoch:
            agents = [
                Agent(env, DQN_HYPERPARAMS, device, 'first_0', agent1_dir + 'epoch_{}.pt'.format(current_player-20)),
                Agent(env, DQN_HYPERPARAMS, device, 'second_0', agent2_dir + 'epoch_{}.pt'.format(current_player))
                ]
            r = []
            for num in range(eval_num):
                print('adjacent ,current_player:', current_player, 'eval_num:', num)
                env.reset()
                env.step(4)
                env.step(4)

                for i in range(MAX_STEP):
                    for n in range(2):
                        obs, reward, done, trunc, info = env.last()
                        action = agents[n].select_greedy_action(obs)
                        env.step(action)
                        if(env.last()[1] > 0):
                            r.append(n)

                    if reward > 0 or reward < 0 and not done and not trunc:
                        env.step(4)
                        env.step(4)

                    if done or trunc:
                        if trunc:
                            r.append(-1)
                        break
            print('adjacent, current_player:', current_player, 'mean:', np.mean(r))
            adjacent_r.append(r)
            current_player += interval

        return adjacent_r

    fixed = fixed_opponent()
    adjacent = adjacent_opponent()
    return fixed, adjacent

def process(agent1_dir, agent2_dir):
    data = evaluation(agent1_dir, agent2_dir)

    f1 = open('fixed.csv', 'w')
    f1.write(str(data[0]).replace('[[', '').replace('[', '\n').replace(']', ''))
    f1.close()

    f2 = open('adjacent.csv', 'w')
    f2.write(str(data[1]).replace('[[', '').replace('[', '\n').replace(']', ''))
    f2.close()

    fixed_mean = []
    for n in data[0]:
        fixed_mean.append(np.mean(n))

    adjacent_mean = []
    for n in data[1]:
        adjacent_mean.append(np.mean(n))

    fixed_mean = np.array(fixed_mean)
    adjacent_mean = np.array(adjacent_mean)



    plt.plot([i*20 for i in range(len(fixed_mean))], fixed_mean)
    plt.xlabel('model')
    plt.ylabel('mean_reward')
    plt.title('evaluating with fixed opponent')
    plt.savefig('fixed.png')

    plt.plot([i*20 for i in range(len(adjacent_mean))], adjacent_mean)
    plt.xlabel('model')
    plt.ylabel('mean_reward')
    plt.title('evaluating with adjacent opponent')
    plt.savefig('adjacent.png')

process(agent1_dir, agent2_dir)




