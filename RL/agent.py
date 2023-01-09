class Agent:

    def __init__(self, lr: float, y: float) -> None:
        self.model = None
        self.lr = lr
        self.y = y
        self.e = 1
        self.e_min = 0.01
        self.train = True
        self.reward_history = {'step': [],
                               'epis': []}
        self.step_count = 0
        self.episode_count = 0

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

    def decay_epsilon(self, rate=0.999999):
        self.e = max(self.e_min, self.e * rate)
