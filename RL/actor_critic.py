import torch
from .deep_agent import DeepAgent
import numpy as np


class ActorCriticAgent(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.log_probs = []
        self.values = []
        self.eps = np.finfo(np.float32).eps.item()
        self.loss_fn = torch.nn.HuberLoss()

    def policy(self, state):
        self.step_count += 1
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action, log_prob, value = self.model(state)
        if self.train:
            self.log_probs.append(log_prob)
            self.values.append(value)
        return action

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, episode_over: bool):
        if self.train:
            self.rewards.append(reward)
            if episode_over:
                self.episode_count += 1
                self.reward_history.append(np.sum(self.rewards))
                self.update_model()
                print(f"Episode: {self.episode_count} | Train: {self.train_count} | r: {self.reward_history[-1]:.6f}")

    def update_model(self):
        self.train_count += 1
        G = []
        r_sum = 0
        for r in reversed(self.rewards):
            r_sum = r_sum * self.y + r
            G.append(r_sum)
        G = torch.tensor(list(reversed(G)), dtype=torch.float32).to(self.device)
        G -= G.mean()
        if len(G) > 1:
            G /= (G.std() + self.eps)

        V = torch.cat(self.values)

        with torch.no_grad():
            A = G - V

        actor_loss = torch.stack([-log_prob * a for log_prob, a in zip(self.log_probs, A)]).sum()
        critic_loss = self.loss_fn(V, G)

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.rewards = []
        self.log_probs = []
        self.values = []
