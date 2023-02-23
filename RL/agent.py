class Agent:

    def __init__(self, state_space_size: int, action_space_size: int) -> None:
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.lr = 0.001
        self.y = 0.99
        self.e = 1
        self.e_min = 0.01
        self.e_decay = 0.999999
        self.model = None
        self.train = True
        self.step_count = 0
        self.episode_count = 0
        self.train_count = 0

    def create_model(self, *args, **kwargs) -> None:
        pass

    def save_model(self, path) -> None:
        pass

    def load_model(self, path) -> None:
        pass

    def learn(self, *args, **kwargs) -> None:
        pass

    def policy(self, state, greedy=False):
        pass

    def decay_epsilon(self, rate=None):
        self.e = max(self.e_min, self.e * rate) if rate else max(self.e_min, self.e * self.e_decay)
