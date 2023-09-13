import torch
from torch.distributions import Categorical
from .deep_agent import DeepAgent
import numpy as np


class OneStepActorCriticAgent(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.actor = None
        self.critic = None
        self.eps = np.finfo(np.float32).eps.item()
        self.loss_fn = torch.nn.HuberLoss(reduction="mean")
        self.i = 1
        self.reward_norm_factor = 1.0
        self.entropy_coef = 0.1
        del self.model
        del self.lr

    def create_model(self,
                     actor: torch.nn.Module,
                     critic: torch.nn.Module,
                     actor_lr: float,
                     critic_lr: float,
                     gamma: float = 0.99,
                     entropy_coef: float = 0.01,
                     reward_norm_factor: float = 1.0):
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.reward_norm_factor = reward_norm_factor
        self.actor = actor(self.state_space_size, self.action_space_size)
        self.actor.to(self.device)
        self.actor.train()
        self.critic = critic(self.state_space_size)
        self.critic.to(self.device)
        self.critic.train()
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr}
        ])

    @torch.no_grad()
    def policy(self, state: np.ndarray):
        self.step_counter += 1
        self.actor.eval()

        if state.ndim == 1:
            state = torch.Tensor(state).unsqueeze(0).to(self.device)
        else:
            state = torch.Tensor(state).to(self.device)

        probs = self.actor(state).squeeze(0)
        distribution = Categorical(probs)
        action = distribution.sample()

        return int(action.cpu().numpy())

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, done: bool):
        self.rewards.append(reward)
        self.update_model(state, action, next_state, reward, done)
        if done:
            self.i = 1
            self.episode_counter += 1
            self.step_counter = 0
            self.reward_history.append(np.sum(self.rewards))
            self.rewards.clear()
            print(f"Episode: {self.episode_counter} | Train: {self.train_counter} | r: {self.reward_history[-1]:.6f}")

    def update_model(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, done: bool):
        self.train_counter += 1
        self.actor.train()
        self.critic.train()
        self.optimizer.zero_grad()

        state = torch.Tensor(state).unsqueeze(0).to(self.device) if state.ndim == 1 else torch.Tensor(state).to(self.device)
        next_state = torch.Tensor(next_state).unsqueeze(0).to(self.device) if next_state.ndim == 1 else torch.Tensor(next_state).to(self.device)
        action = torch.Tensor([action]).to(self.device)

        reward /= self.reward_norm_factor

        # Bug? It doesn't seem to need to compute computational graph when forwarding next_state.
        # But skipping that part with torch.no_grad() breaks learning. Weird!

        # Next state value
        with torch.no_grad():
            V_ = (1.0 - done) * self.critic(next_state)

        # Current state value
        V = self.critic(state)

        # Expected return from current state
        G = reward + self.gamma * V_

        # TD error/Advantage
        A = G - (1.0 - done) * V.detach()

        critic_loss = A * self.loss_fn(V, G)

        distribution = Categorical(probs=self.actor(state))
        LOG = distribution.log_prob(action)
        ENTROPY = distribution.entropy() * self.entropy_coef

        actor_loss = A * LOG

        loss = -self.i * (actor_loss - critic_loss + ENTROPY)

        loss.backward()
        self.optimizer.step()

        self.i *= self.gamma
