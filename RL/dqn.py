import torch
import numpy as np
from .deep_agent import DeepAgent
from .utils import ReplayBufferBase


class DeepQNetworkAgent(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.e = 1
        self.e_min = 0.01
        self.e_decay = 0.999999
        self.target_model = None
        self.buffer = None
        self.batch = 0
        self.reward_norm_factor = 1.0
        self.target_update_freq = 0
        self.target_update_rate = 0
        self.target_update_method = "soft"
        self.target_update_fn = self.target_update_soft
        self.loss_fn = torch.nn.HuberLoss()

    def create_buffer(self, buffer: ReplayBufferBase):
        if buffer.min_size == 0:
            buffer.min_size = self.batch
        self.buffer = buffer

    def create_model(self, model: torch.nn.Module, lr: float, y: float, e_decay: float = 0.999999, batch: int = 64, target_update_method: str = "soft", tuf: int = 10, tau: float = 0.001, reward_norm_factor: float = 1.0):
        super().create_model(model, lr, y)
        self.target_model = model(self.state_space_size, self.action_space_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.to(self.device)
        self.target_model.eval()
        self.e_decay = e_decay
        self.batch = batch
        self.target_update_freq = tuf
        self.target_update_rate = tau
        self.target_update_method = target_update_method
        if self.target_update_method == "hard":
            self.target_update_fn = self.target_update_hard
        self.reward_norm_factor = reward_norm_factor

    def load_model(self, path) -> None:
        super().load_model(path)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.to(self.device)
        self.target_model.eval()

    @torch.no_grad()
    def policy(self, state: np.ndarray):
        self.step_count += 1
        self.model.eval()
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if self.training and np.random.random() < self.e:
            return np.random.choice(self.action_space_size)
        else:
            return torch.argmax(self.model(state)).item()

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, episode_over: bool):
        """update: ['hard', 'soft'] = 'soft'"""
        self.buffer.push(state, action, next_state, reward, episode_over)
        if self.buffer.trainable:
            self.rewards.append(reward)
            self.update_model()
            self.target_update_fn()
            if episode_over:
                self.decay_epsilon()
                self.episode_count += 1
                self.step_count = 0
                self.reward_history.append(np.sum(self.rewards))
                self.rewards.clear()
                print(f"Episode: {self.episode_count} | Train: {self.train_count} | e: {self.e:.6f} | r: {self.reward_history[-1]:.6f}")

    def decay_epsilon(self):
        self.e = max(self.e_min, self.e * self.e_decay)

    def target_update_hard(self):
        if self.train_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def target_update_soft(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_((1.0 - self.target_update_rate) * target_param.data + self.target_update_rate * local_param.data)

    def update_model(self):
        self.train_count += 1
        s, a, ns, r, d = self.buffer.sample(self.batch)
        r /= self.reward_norm_factor
        states = torch.tensor(s).float().to(self.device)
        next_states = torch.tensor(ns).float().to(self.device)
        r = torch.tensor(r).float().to(self.device)
        d = torch.tensor(d).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            current_qs = self.model(states)
            future_qs = self.target_model(next_states)
            current_qs[torch.arange(self.batch), a] = r + (1 - d) * self.y * torch.max(future_qs, dim=1).values

        self.model.train()
        preds = self.model(states)
        loss = self.loss_fn(preds, current_qs).to(self.device)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
