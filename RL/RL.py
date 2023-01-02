import numpy as np
from matplotlib import pyplot as plt


class Agent:

    def __init__(self, lr, y, e_min=0.01) -> None:
        self.model = None
        self.lr = lr
        self.y = y
        self.e = 1
        self.E_min = e_min
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
        self.e = max(self.E_min, self.e*rate)


class QLAgent(Agent):

    def __init__(self, lr=0.1, y=0.99, e_min=0.01) -> None:
        super().__init__(lr, y, e_min)
        self.__action_space_size = 0

    def create_model(self, dim: tuple) -> None:
        self.model = np.zeros(dim)
        self.__action_space_size = dim[-1]

    def save_model(self, path) -> None:
        np.save(path, self.model)

    def load_model(self, path) -> None:
        self.model = np.load(path)

    def learn(self, s: tuple, a: int, r: float, ns: tuple, d: bool) -> None:
        self.reward_history['step'].append(r)
        if not d:
            max_future_q_value = np.max(self.model[ns])
            current_q_value = self.model[s][a]
            new_q_value = current_q_value + self.lr * \
                (r + self.y*max_future_q_value - current_q_value)
            self.model[s][a] = new_q_value
        else:
            self.model[s][a] = r
            avg = sum(self.reward_history['step'][-self.step_count:])/self.step_count
            self.reward_history['epis'].append(avg)
            self.reward_history['step'].clear()
            self.episode_count += 1

    def policy(self, state, use_e=False):
        self.step_count += 1
        if use_e and np.random.random() < self.e:
            return np.random.randint(0, self.__action_space_size)
        return np.argmax(self.model[state])

    def plot(self):
        plt.plot(self.reward_history['epis'])
        plt.show()
