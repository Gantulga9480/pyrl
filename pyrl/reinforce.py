import torch
from torch.distributions import Categorical
from .deep_agent import DeepAgent
import numpy as np


class ReinforceAgent(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.eps = np.finfo(np.float32).eps.item()
        self.reward_norm_factor = 1.0
        self.entropy_coef = 0.1
        self.state_buffer = []
        self.action_buffer = []

    def create_model(self,
                     model: torch.nn.Module,
                     lr: float,
                     gamma: float,
                     entropy_coef: float,
                     reward_norm_factor: float = 1.0):
        super().create_model(model, lr, gamma)
        self.entropy_coef = entropy_coef
        self.reward_norm_factor = reward_norm_factor

    @torch.no_grad()
    def policy(self, state: np.ndarray):
        self.step_counter += 1
        self.model.eval()

        if state.ndim == 1:
            state = torch.Tensor(state).unsqueeze(0).to(self.device)
        else:
            state = torch.Tensor(state).to(self.device)

        probs = self.model(state).squeeze(0)
        distribution = Categorical(probs)
        action = distribution.sample()

        return int(action.cpu().numpy())

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, done: bool):
        self.rewards.append(reward)
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        if done:
            if len(self.rewards) > 1:
                self.episode_counter += 1
                self.update_model()
                self.reward_history.append(np.sum(self.rewards))
                print(f"Episode: {self.episode_counter} | Train: {self.train_counter} | r: {self.reward_history[-1]:.6f}")
            self.rewards.clear()
            self.state_buffer.clear()
            self.action_buffer.clear()
            self.step_counter = 0

    def update_model(self):
        self.train_counter += 1
        self.model.train()
        self.optimizer.zero_grad()

        b_states = torch.tensor(np.array(self.state_buffer)).float().to(self.device)
        b_actions = torch.tensor(self.action_buffer).to(self.device)
        b_rewards = torch.tensor(self.rewards).float().to(self.device)
        b_rewards /= self.reward_norm_factor

        # Compute return G
        G = self.VAE(b_rewards)
        # Norm G
        G -= G.mean()
        G /= (G.std() + self.eps)
        # Compute log prob and entropy
        distribution = Categorical(probs=self.model(b_states))
        LOG = distribution.log_prob(b_actions)
        ENTROPY = distribution.entropy() * self.entropy_coef
        # Compute policy loss + entropy bonus + gradient ascent instead of descent
        loss = -(LOG * G + ENTROPY).mean()
        loss.backward()
        # Optimize model parameters
        self.optimizer.step()

    def VAE(self, rewards):
        reward_sum = 0
        for i in reversed(range(rewards.size()[0])):
            rewards[i] = reward_sum = reward_sum * self.gamma + rewards[i]
        return rewards
