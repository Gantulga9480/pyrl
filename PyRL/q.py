import numpy as np
from .agent import Agent


class QLearningAgent(Agent):

    def __init__(self, state_space_size: int, action_space_size: int) -> None:
        super().__init__(state_space_size, action_space_size)
        self.e = 1
        self.e_min = 0.01
        self.e_decay = 0.999999

    def create_model(self, lr: float = 0.1, gamma: float = 0.9, e_decay: float = 0.999) -> None:
        self.lr = lr
        self.gamma = gamma
        self.e_decay = e_decay
        if isinstance(self.state_space_size, tuple):
            self.model = np.zeros((*self.state_space_size, self.action_space_size))
        elif isinstance(self.state_space_size, int):
            self.model = np.zeros((self.state_space_size, self.action_space_size))
        else:
            self.model = np.zeros((self.state_space_size, self.action_space_size))

    def save_model(self, path) -> None:
        np.save(path, self.model)

    def load_model(self, path) -> None:
        self.model = np.load(path)

    def learn(self, state: tuple, action: int, next_state: tuple, reward: float, done: bool) -> None:
        self.rewards.append(reward)
        self.train_count += 1
        if not done:
            max_future_q_value = np.max(self.model[next_state])
            current_q_value = self.model[state][action]
            new_q_value = current_q_value + self.lr * (reward + self.y * max_future_q_value - current_q_value)
            self.model[state][action] = new_q_value
        else:
            self.model[state][action] = reward
            self.episode_counter += 1
            self.step_counter = 0
            self.reward_history.append(np.sum(self.rewards))
            self.rewards.clear()
            print(f"Episode: {self.episode_counter} | Train: {self.train_count} | e: {self.e:.6f} | r: {self.reward_history[-1]:.6f}")
        self.decay_epsilon()

    def policy(self, state):
        self.step_counter += 1
        if self.training and np.random.random() < self.e:
            return np.random.choice(self.action_space_size)
        return np.argmax(self.model[state])

    def decay_epsilon(self):
        self.e = max(self.e_min, self.e * self.e_decay)
