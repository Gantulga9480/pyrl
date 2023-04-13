import torch
import numpy as np
from .deep_agent import DeepAgent
from .utils import ReplayBufferBase


class DeepDeterministicPolicyGradientAgent(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.actor = None
        self.target_actor = None
        self.critic = None
        self.target_critic = None
        self.buffer = None
        self.batch = 0
        self.target_update_rate = 0
        self.noise = 0
        self.train_count = 0
        self.loss_fn = torch.nn.HuberLoss()
        self.reward_norm_factor = 1.0
        del self.lr
        del self.model
        del self.optimizer

    def create_buffer(self, buffer: ReplayBufferBase):
        if buffer.min_size == 0:
            buffer.min_size = self.batch
        self.buffer = buffer

    def create_model(self, actor: torch.nn.Module, critic: torch.nn.Module, actor_lr: float, critic_lr: float, gamma: float, noise_std: float, batch: int = 64, tau: float = 0.001, reward_norm_factor: float = 1.0):
        self.gamma = gamma
        self.noise_std = noise_std
        self.batch = batch
        self.target_update_rate = tau
        self.reward_norm_factor = reward_norm_factor
        self.actor = actor(self.state_space_size, self.action_space_size)
        self.target_actor = actor(self.state_space_size, self.action_space_size)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor.to(self.device)
        self.actor.train()
        self.target_actor.to(self.device)
        self.target_actor.eval()
        self.critic = critic(self.state_space_size, self.action_space_size)
        self.target_critic = critic(self.state_space_size, self.action_space_size)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic.to(self.device)
        self.critic.train()
        self.target_critic.to(self.device)
        self.target_critic.eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def policy(self, state):
        self.step_counter += 1
        self.actor.eval()
        state = torch.Tensor(state).to(self.device)
        action = self.actor(state).cpu().numpy()
        if self.training:
            return (action + np.random.normal(0, self.noise_std)).clip(-1, 1)
        else:
            return action

    def learn(self, state: np.ndarray, action: float, next_state: np.ndarray, reward: float, done: bool):
        self.buffer.push(state, action, next_state, reward, done)
        if self.buffer.trainable:
            self.rewards.append(reward)
            self.update_model()
            self.update_target()
            if done:
                self.step_counter = 0
                self.episode_counter += 1
                self.reward_history.append(np.sum(self.rewards))
                self.rewards.clear()
                print(f"Episode: {self.episode_counter} | Train: {self.train_count} | r: {self.reward_history[-1]:.6f}")

    def update_target(self):
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.target_update_rate * local_param.data + (1.0 - self.target_update_rate) * target_param.data)

        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.target_update_rate * local_param.data + (1.0 - self.target_update_rate) * target_param.data)

    def update_model(self):
        self.train_count += 1
        s, a, ns, r, d = self.buffer.sample(self.batch)
        r /= self.reward_norm_factor
        states = torch.tensor(s).float().to(self.device)
        actions = torch.tensor(a).float().view(self.batch, self.action_space_size).to(self.device)
        next_states = torch.tensor(ns).float().to(self.device)
        rewards = torch.tensor(r).float().view(self.batch, 1).to(self.device)
        dones = torch.tensor(d).float().view(self.batch, 1).to(self.device)

        with torch.no_grad():
            y = rewards + (1 - dones) * self.gamma * self.target_critic(next_states, self.target_actor(next_states))

        self.actor.train()
        preds = self.critic(states, actions)
        critic_loss = self.loss_fn(preds, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        p_actions = self.actor(states)
        actor_loss = -self.critic(states, p_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
