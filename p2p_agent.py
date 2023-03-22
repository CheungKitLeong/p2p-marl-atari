from agent import Agent
from agent_control import AgentControl
from neural_net import DQN
import torch.optim as optim
import numpy as np
import torch


class P2PAgentControl(AgentControl):
    def __init__(self, env, device, lr, gamma, name, pred_lr=1e-4):
        super().__init__(env, device, lr, gamma, name)
        self.predictor_nn = DQN(input_shape=env.observation_space(name).shape,
                                num_of_actions=env.action_space(name).n).to(device)
        self.pred_optimizer = optim.AdamW(self.predictor_nn.parameters(), lr=pred_lr)

    def improve(self, mini_batch):
        loss = super().improve(mini_batch)
        self.update_predictor(mini_batch)
        return loss

    def update_predictor(self, mini_batch):
        states, actions, next_states, rewards, dones = mini_batch
        # Transform numpy array to Tensor and send it to GPU
        states_tensor = torch.as_tensor(states, dtype=torch.float).to(self.device)

        predictor_action_value = self.predictor_nn(states_tensor)
        moving_action_value = self.moving_nn(states_tensor).detach()
        # loss_fn = nn.MSELoss(reduction=None)
        predictor_loss = self.loss(predictor_action_value, moving_action_value)

        self.pred_optimizer.zero_grad()
        predictor_loss.backward()
        self.pred_optimizer.step()


class P2PAgent(Agent):

    def __init__(self, env, hyperparameters, device, name, load_path=None):
        super().__init__(env, hyperparameters, device, name, load_path=load_path)
        self.advisors = None
        self.agent_control = P2PAgentControl(env, device, hyperparameters['learning_rate'], hyperparameters['gamma'],
                                             name)
        self.ask_threshold = hyperparameters['ask_threshold']
        self.give_threshold = hyperparameters['give_threshold']
        self.ask_budget = hyperparameters['ask_budget']
        self.give_budget = hyperparameters['give_budget']

    def set_advisor(self, advisors):
        """Set the adviser agent
        advisor: list of agent objects
        """
        self.advisors = advisors

        # In training function:
        # agents = [P2P_Agent(env, hyperparams, device, 'first_0'), P2P_Agent(env, hyperparams, device, 'second_0')]
        # agents[0].set_advisor([agents[1]])
        # agents[1].set_advisor([agents[0]])

    def compute_uncertainty(self, obs):
        tensor_obs = torch.tensor(np.array([obs]), dtype=torch.float).to(self.device)
        predictor_action_value = self.predictor_nn(tensor_obs).detach()
        moving_action_value = self.moving_nn(tensor_obs).detach()
        uncertainty = self.loss(predictor_action_value, moving_action_value)
        return uncertainty.item()

    def check_ask(self, obs) -> bool:
        """Determine should the agent ask advice"""
        return self.compute_uncertainty(obs) > self.ask_threshold

    def check_advise(self, obs) -> bool:
        """Determine should the agent give advise"""
        return self.compute_uncertainty(obs) < self.give_threshold

    def ask_advice(self, obs):
        action = None
        advices = []
        for advisor in self.advisors:
            a = advisor.give_advice(obs)
            if a is not None:
                advices.append(a)

        if not advices:  # check if the list is empty
            action = max(set(advices), key=advices.count)  # Majority voting

        return action

    def select_eps_greedy_action(self, obs):
        """Add ask advice features"""
        action = None
        if self.ask_budget > 0 and self.check_ask():
            action = self.ask_advice(obs)
        if action is None:
            action = super().select_eps_greedy_action(obs)
        else:
            self.ask_budget -= 1

        return action

    def give_advice(self, obs):
        """Called when other agents ask me advice"""
        action = None
        if self.give_budget > 0 and self.check_advise(obs):
            action = self.select_greedy_action(obs)
            self.give_budget -= 1

        return action
