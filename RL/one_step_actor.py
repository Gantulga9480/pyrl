import torch
from torch.distributions import Categorical
import numpy as np
from .deep_agent import DeepAgent


class OneStepActor(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.log_prob = None
        self.eps = np.finfo(np.float32).eps.item()
        self.reward_norm_factor = 1.0
        self.g = 0.0
        self.i = 1.0

    def create_model(self, model: torch.nn.Module, lr: float, y: float, reward_norm_factor: float = 1.0):
        self.reward_norm_factor = reward_norm_factor
        return super().create_model(model, lr, y)

    def policy(self, state):
        self.step_count += 1
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if not self.train:
            self.model.eval()
        probs = self.model(state)
        distribution = Categorical(probs)
        action = distribution.sample()
        if self.train:
            self.log_prob = distribution.log_prob(action)
        return action.item()

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, episode_over: bool):
        self.rewards.append(reward)
        if self.train:
            self.update_model(reward)
        if episode_over:
            self.i = 1.0
            self.g = 0.0
            self.episode_count += 1
            self.step_count = 0
            self.reward_history.append(np.sum(self.rewards))
            self.rewards.clear()
            print(f"Episode: {self.episode_count} | Train: {self.train_count} | r: {self.reward_history[-1]:.6f}")

    def update_model(self, reward):
        self.train_count += 1
        self.model.train()
        td_error = reward + self.y * self.g - self.g
        self.loss = -self.log_prob * (td_error * self.i)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.g = reward + self.y * self.g
        self.i *= self.y
