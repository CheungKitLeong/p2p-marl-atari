from agent import Agent
import torch


def train_basic(env, hyperparams, num_of_episode, MAX_STEP=10000, BATCH_SIZE=32):
    """The very first training loop, no force fire, no action mapping, no self play"""
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # Creating agents
    hyperparams['mode'] = "basic"
    agents = [Agent(env, hyperparams, device, 'first_0'), Agent(env, hyperparams, device, 'second_0')]

    for episode in range(num_of_episode + 1):
        # Run one episode
        # Reset the environment and get the initial observation
        env.reset()
        old_obs = [None] * 2
        old_action = [0] * 2
        for i in range(MAX_STEP):
            # Inside a episode
            done = False
            trunc = False

            for n in range(2):
                new_obs, reward, done, trunc, info = env.last()
                action = agents[n].select_eps_greedy_action(new_obs)
                env.step(action)
                if old_obs[n] is not None:
                    agents[n].add_to_buffer(old_obs[n], old_action[n], new_obs, reward, (done or trunc))
                agents[n].sample_and_improve(BATCH_SIZE)
                old_obs[n] = new_obs
                old_action[n] = action

            # If the game is over, reset the environment
            if done or trunc:
                break
        # Call when end a episode
        agents[0].print_info()
        agents[1].print_info()
        agents[0].reset_parameters()
        agents[1].reset_parameters()


def train_stationary(env, hyperparams, num_of_episode, MAX_STEP=10000, BATCH_SIZE=32, episode_to_switch=50):
    """Fixed one agent when training other agent"""
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    hyperparams['mode'] = "stationary"

    # Creating agents
    agents = [Agent(env, hyperparams, device, 'first_0'), Agent(env, hyperparams, device, 'second_0')]

    for episode in range(num_of_episode * 2):
        training_agent = 0 if episode % (episode_to_switch * 2) < episode_to_switch else 1
        env.reset()
        old_obs = [None] * 2
        old_action = [0] * 2
        for i in range(MAX_STEP):
            done = False
            trunc = False

            for n in range(2):
                obs, reward, done, trunc, info = env.last()
                if n == training_agent:
                    new_obs, reward, done, trunc, info = env.last()
                    action = agents[n].select_eps_greedy_action(new_obs)
                    env.step(action)
                    if old_obs[n] is not None:
                        agents[n].add_to_buffer(old_obs[n], old_action[n], new_obs, reward, (done or trunc))
                    agents[n].sample_and_improve(BATCH_SIZE)
                    old_obs[n] = new_obs
                    old_action[n] = action
                else:
                    # Non-training agent
                    action = agents[n].select_greedy_action(obs)
                    env.step(action)

            if done or trunc:
                break

        for n in range(2):
            if n == training_agent:
                agents[n].print_info()
                agents[n].reset_parameters()

    # Run one more episode to save the final model
    env.reset()
    old_obs = [None] * 2
    old_action = [0] * 2
    for i in range(MAX_STEP):
        done = False
        trunc = False

        for n in range(2):
            new_obs, reward, done, trunc, info = env.last()
            action = agents[n].select_eps_greedy_action(new_obs)
            env.step(action)
            if old_obs[n] is not None:
                agents[n].add_to_buffer(old_obs[n], old_action[n], new_obs, reward, (done or trunc))
            agents[n].sample_and_improve(BATCH_SIZE)
            old_obs[n] = new_obs
            old_action[n] = action

        if done or trunc:
            break

    agents[0].print_info()
    agents[1].print_info()
    agents[0].reset_parameters()
    agents[1].reset_parameters()
