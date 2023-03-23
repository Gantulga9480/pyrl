import torch
from torch.distributions import Categorical
from .deep_agent import DeepAgent
import numpy as np


class ReinforceAgent(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.log_probs = []
        self.eps = np.finfo(np.float32).eps.item()
        self.reward_norm_factor = 1.0

    def create_model(self, model: torch.nn.Module, lr: float, y: float, reward_norm_factor: float = 1.0):
        self.reward_norm_factor = reward_norm_factor
        return super().create_model(model, lr, y)

    def policy(self, state):
        self.step_count += 1
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if not self.train:
            self.model.eval()
            with torch.no_grad():
                probs = self.model(state).squeeze(0)
                distribution = Categorical(probs)
                action = distribution.sample()
                return action.item()
        probs = self.model(state).squeeze(0)
        distribution = Categorical(probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).unsqueeze(0)
        self.log_probs.append(log_prob)
        return action.item()

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, episode_over: bool):
        self.rewards.append(reward)
        if episode_over:
            self.step_count = 0
            self.reward_history.append(np.sum(self.rewards))
            if len(self.rewards) > 1:
                self.episode_count += 1
                self.update_model()
                print(f"Episode: {self.episode_count} | Train: {self.train_count} | r: {self.reward_history[-1]:.6f}")
            self.rewards.clear()

    def update_model(self):
        self.train_count += 1
        self.model.train()
        g = np.array(self.rewards, dtype=np.float32)
        g /= self.reward_norm_factor
        r_sum = 0
        for i in reversed(range(g.shape[0])):
            g[i] = r_sum = r_sum * self.y + g[i]
        G = torch.tensor(g).to(self.device)
        G -= G.mean()
        G /= (G.std() + self.eps)

        log_probs = torch.cat(self.log_probs)
        loss = -log_probs @ G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
