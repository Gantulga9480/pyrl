import torch
import numpy as np
from .deep_agent import DeepAgent
from .utils import ReplayBufferBase


class DeepQNetworkAgent(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.target_model = None
        self.buffer = None
        self.batchs = 0
        self.target_update_freq = 0
        self.target_update_rate = 0
        self.loss_fn = torch.nn.HuberLoss()

    def create_buffer(self, buffer: ReplayBufferBase):
        if buffer.min_size == 0:
            buffer.min_size = self.batchs
        self.buffer = buffer

    def create_model(self, model: torch.nn.Module, lr: float, y: float, e_decay: float = 0.999999, batchs: int = 64, target_update_freq: int = 10, tau: float = 0.001):
        super().create_model(model, lr, y)
        self.target_model = model(self.state_space_size, self.action_space_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.to(self.device)
        self.target_model.eval()
        self.e_decay = e_decay
        self.batchs = batchs
        self.target_update_freq = target_update_freq
        self.target_update_rate = tau

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
        is_batch = len(state.size()) > 1
        if not is_batch:
            if self.train and np.random.random() < self.e:
                return np.random.choice(list(range(self.action_space_size)))
            else:
                return torch.argmax(self.model(state)).item()
        else:
            if self.train and np.random.random() < self.e:
                return [np.random.choice(list(range(self.action_space_size))) for _ in range(len(state))]
            else:
                return torch.argmax(self.model(state), axis=1).tolist()

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, episode_over: bool, update: str = "soft"):
        """update: ['hard', 'soft'] = 'soft'"""
        batch = len(state.shape) > 1
        if not batch:
            self.rewards.append(reward)
            self.buffer.push(state, action, next_state, reward, episode_over)
        else:
            self.rewards.append(np.mean(reward))
            self.buffer.extend(state, action, next_state, reward, episode_over)
        if self.buffer.trainable and self.train:
            self.update_model()
            if update == "soft":
                self.target_update_soft()
            elif update == "hard":
                self.target_update_hard()
            else:
                raise ValueError(f"Wrong target update mode -> {update}")
            self.decay_epsilon()
        if episode_over:
            self.episode_count += 1
            self.reward_history.append(np.sum(self.rewards))
            self.rewards = []
            print(f"Episode: {self.episode_count} | Train: {self.train_count} | e: {self.e:.6f} | r: {self.reward_history[-1]:.6f}")

    def target_update_hard(self):
        if self.train_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def target_update_soft(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.target_update_rate * local_param.data + (1.0 - self.target_update_rate) * target_param.data)

    def update_model(self):
        self.train_count += 1
        s, a, ns, r, d = self.buffer.sample(self.batchs)
        self.model.eval()
        states = torch.tensor(s, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(ns, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            current_qs = self.model(states)
            future_qs = self.target_model(next_states)
            for i in range(len(s)):
                current_qs[i][a[i]] = (r[i] + (1 - d[i]) * self.y * torch.max(future_qs[i])).item()

        self.model.train()
        preds = self.model(states)
        loss = self.loss_fn(preds, current_qs).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
