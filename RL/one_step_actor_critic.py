import torch
from torch.distributions import Categorical
from .deep_agent import DeepAgent
import numpy as np


class OneStepActorCriticAgent(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.actor = None
        self.critic = None
        self.LOG = None
        self.ENTROPY = None
        self.eps = np.finfo(np.float32).eps.item()
        self.loss_fn = torch.nn.HuberLoss(reduction="mean")
        self.i = 1
        self.reward_norm_factor = 1.0
        self.entropy_coef = 0.1
        del self.model
        del self.optimizer
        del self.lr

    def create_model(self,
                     actor: torch.nn.Module,
                     critic: torch.nn.Module,
                     actor_lr: float,
                     critic_lr: float,
                     gamma: float,
                     entropy_coef: float,
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
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def policy(self, state):
        self.step_counter += 1
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        if not self.training:
            self.actor.eval()
            with torch.no_grad():
                probs = self.actor(state)
                distribution = Categorical(probs)
                action = distribution.sample()
            return action.item()
        self.actor.train()
        probs = self.actor(state)
        distribution = Categorical(probs)
        action = distribution.sample()
        self.LOG = distribution.log_prob(action)
        self.ENTROPY = distribution.entropy()
        return action.item()

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, done: bool):
        self.rewards.append(reward)
        self.update_model(state, next_state, reward, done)
        if done:
            self.i = 1
            self.episode_counter += 1
            self.step_counter = 0
            self.reward_history.append(np.sum(self.rewards))
            self.rewards.clear()
            print(f"Episode: {self.episode_counter} | Train: {self.train_count} | r: {self.reward_history[-1]:.6f}")

    def update_model(self, state, next_state, reward, done):
        self.train_count += 1

        reward /= self.reward_norm_factor
        state = torch.tensor(state).float().to(self.device)
        next_state = torch.tensor(next_state).float().to(self.device)

        # Bug? It doesn't seem to need to compute computational graph when forwarding next_state.
        # But skipping that part with torch.no_grad() breaks learning. Weird!

        # Next state value
        V_ = (1.0 - done) * self.critic(next_state)

        # Current state value
        V = self.critic(state)

        # Expected return from current state
        G = reward + self.gamma * V_.detach()

        # TD error/Advantage
        A = G.detach() - (1.0 - done) * V.detach()

        critic_loss = A * self.i * self.loss_fn(V, G)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = A * self.i * -self.LOG

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.i *= self.gamma
