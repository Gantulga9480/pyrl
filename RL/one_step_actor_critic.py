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
        self.eps = np.finfo(np.float32).eps.item()
        self.loss_fn = torch.nn.HuberLoss(reduction="sum")
        self.i = 1
        self.reward_norm_factor = 1.0
        del self.model
        del self.optimizer
        del self.lr

    def create_model(self, actor: torch.nn.Module, critic: torch.nn.Module, actor_lr: float, critic_lr: float, y: float, reward_norm_factor: float = 1.0):
        self.y = y
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
        self.step_count += 1
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        if not self.training:
            self.actor.eval()
            with torch.no_grad():
                probs = self.actor(state)
                distribution = Categorical(probs)
                action = distribution.sample()
            return action.item()
        probs = self.actor(state)
        distribution = Categorical(probs)
        action = distribution.sample()
        self.LOG = distribution.log_prob(action)
        return action.item()

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, episode_over: bool):
        self.rewards.append(reward)
        self.update_model(state, next_state, reward, episode_over)
        if episode_over:
            self.i = 1
            self.episode_count += 1
            self.step_count = 0
            self.reward_history.append(np.sum(self.rewards))
            self.rewards.clear()
            print(f"Episode: {self.episode_count} | Train: {self.train_count} | r: {self.reward_history[-1]:.6f}")

    def update_model(self, state, next_state, reward, done):
        self.train_count += 1
        self.actor.train()

        reward /= self.reward_norm_factor
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

        # Bug? It doesn't seem to need to compute computational graph when forwarding next_state.
        # But skipping that part breaks learning. Weird!
        # with torch.no_grad():
        # Next state value
        V_ = (1.0 - done) * self.critic(next_state)
        # Current state value
        V = self.critic(state)

        # Expected return
        G = reward / self.reward_norm_factor + self.y * V_

        critic_loss = self.loss_fn(V, G)
        critic_loss *= self.i

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Swapping position for no negative sign on actor_loss
        # TD error/Advantage
        A = V.item() - G.item()
        actor_loss = self.LOG * A
        actor_loss *= self.i

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.i *= self.y
