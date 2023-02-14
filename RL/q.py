import numpy as np
from .agent import Agent


class QLAgent(Agent):

    def __init__(self, state_space_size: int, action_space_size: int, lr: float, y: float, e_decay: float = 0.99999, seed: int = 1) -> None:
        super().__init__(state_space_size, action_space_size, lr, y, e_decay)
        np.random.seed(seed)
        self.create_model((*self.state_space_size, self.action_space_size))

    def create_model(self, dim: tuple) -> None:
        self.model = np.zeros(dim)

    def save_model(self, path) -> None:
        np.save(path, self.model)

    def load_model(self, path) -> None:
        self.model = np.load(path)

    def learn(self, s: tuple, a: int, ns: tuple, r: float, d: bool) -> None:
        if not d:
            max_future_q_value = np.max(self.model[ns])
            current_q_value = self.model[s][a]
            new_q_value = current_q_value + self.lr * \
                (r + self.y * max_future_q_value - current_q_value)
            self.model[s][a] = new_q_value
        else:
            self.model[s][a] = r * self.y
            self.episode_count += 1
        self.decay_epsilon()

    def policy(self, state, greedy=False):
        self.step_count += 1
        if not greedy and np.random.random() < self.e:
            return np.random.choice(self.action_space_size)
        return np.argmax(self.model[state])
