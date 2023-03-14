from agent import Agent
import torch


def train_basic(env, hyperparams, num_of_episode, MAX_STEP=10000, BATCH_SIZE=32):
    """The very first training loop, no force fire, no action mapping, no self play"""
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Creating agents
    agents = [Agent(env, hyperparams, device, 'first_0'), Agent(env, hyperparams, device, 'second_0')]

    for episode in range(num_of_episode):
        # Run one episode
        # Reset the environment and get the initial observation
        env.reset()

        # Play the game for a few steps with random actions for both players
        for i in range(MAX_STEP):
            done = False
            trunc = False

            for n in range(2):
                obs, reward, done, trunc, info = env.last()
                action = agents[n].select_eps_greedy_action(obs)
                #if reward != 0:
                    #print("%d Agent %d: action: %d, reward: %d" % (i, n, action, reward))
                    #input()
                # print("Agent %d: action: %d" % (n, action))
                env.step(action)
                new_obs = env.last()[0]
                agents[n].add_to_buffer(obs, action, new_obs, reward, (done or trunc))
                agents[n].sample_and_improve(BATCH_SIZE)

            # If the game is over, reset the environment
            if done or trunc:
                break

        agents[0].print_info()
        agents[1].print_info()
        agents[0].reset_parameters()
        agents[1].reset_parameters()