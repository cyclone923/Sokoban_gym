class Memory:
    def __init__(self):
        self.actions = []
        self.obs = []
        self.c_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.obs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]