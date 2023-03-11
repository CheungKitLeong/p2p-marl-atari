from agent import Agent

class P2P_Agent(Agent):

    def __init__(self):
        pass

    def set_advisor(self):
        """Set the adviser agent"""
        pass

    def check_ask(self) -> bool:
        """Determine should the agent ask advice"""
        pass

    def check_advise(self) -> bool:
        """Determine should the agent give advise"""
        pass

    def select_eps_greedy_action(self, obs):
        """Add ask advice features"""
        pass

    def give_advice(self):
        """Called when other agents ask me advice"""
        pass