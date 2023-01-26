import numpy as np
from .agent import Agent


class QLAgent(Agent):

    def __init__(self, action_space: list, lr: float, y: float, e_decay: float = 0.99999) -> None:
        super().__init__(action_space, lr, y, e_decay)

    def create_model(self, dim: tuple) -> None:
        self.model = np.zeros(dim)

    def save_model(self, path) -> None:
        np.save(path, self.model)

    def load_model(self, path) -> None:
        self.model = np.load(path)

    def learn(self, s: tuple, a: int, r: float, ns: tuple, d: bool) -> None:
        if not d:
            max_future_q_value = np.max(self.model[ns])
            current_q_value = self.model[s][a]
            new_q_value = current_q_value + self.lr * \
                (r + self.y * max_future_q_value - current_q_value)
            self.model[s][a] = new_q_value
        else:
            self.model[s][a] = r
            self.episode_count += 1

    def policy(self, state, greedy=False):
        self.step_count += 1
        if not greedy and np.random.random() < self.e:
            return np.random.choice(self.action_space)
        return np.argmax(self.model[state])
