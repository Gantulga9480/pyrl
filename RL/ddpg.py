import torch
from torch import nn
import numpy as np
import os
from .agent import Agent
from .utils import ReplayBufferBase


class DDPGAgent(Agent):

    def __init__(self, state_space_size: int, action_space_size: int, lr: float, y: float, e_decay: float = 0.99999, device: str = 'cpu', seed: int = 1) -> None:
        super(DDPGAgent, self).__init__(state_space_size, action_space_size, lr, y, e_decay)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.target_model = None
        self.model = None
        self.buffer = None
        self.batchs = 0
        self.epochs = 0
        self.device = device
        self.train_freq = 0
        self.update_freq = 0
        self.train_count = 0

    def create_buffer(self, buffer: ReplayBufferBase):
        if buffer.min_size == 0:
            buffer.min_size = self.batchs
        self.buffer = buffer

    def create_model(self, model: torch.nn.Module, target: torch.nn.Module, batchs: int = 64, train_freq: int = 10, update_freq: int = 10):
        self.model = model
        self.target_model = target
        self.target_model.load_state_dict(self.model.state_dict())
        self.model.to(self.device)
        self.model.train()
        self.target_model.to(self.device)
        self.target_model.eval()
        self.batchs = batchs
        self.train_freq = train_freq
        self.update_freq = update_freq
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def save_model(self, path: str) -> None:
        if self.model and path:
            try:
                torch.save(self.model.state_dict(), path)
            except Exception:
                os.makedirs("/".join(path.split("/")[:-1]))
                torch.save(self.model.state_dict(), path)

    def load_model(self, path) -> None:
        try:
            self.model.load_state_dict(torch.load(path))
            self.target_model.load_state_dict(self.model.state_dict())
            self.model.to(self.device)
            self.model.train()
            self.target_model.to(self.device)
            self.target_model.eval()
        except Exception:
            print(f'{path} file not found!')
            exit()

    @torch.no_grad()
    def policy(self, state, greedy=False):
        """greedy - False (default) for training, True for inference"""
        self.step_count += 1
        self.model.eval()
        state = torch.Tensor(state).to(self.device)
        is_batch = len(state.size()) > 1
        if not is_batch:
            if not greedy and np.random.random() < self.e:
                return np.random.choice(list(range(self.action_space_size)))
            else:
                return torch.argmax(self.model(state)).item()
        else:
            if not greedy and np.random.random() < self.e:
                return [np.random.choice(list(range(self.action_space_size))) for _ in range(len(state))]
            else:
                return torch.argmax(self.model(state), axis=1).tolist()

    def learn(self, state, action, next_state, reward, episode_over):
        batch = len(np.array(state).shape) > 1
        if not batch:
            self.buffer.push([state, action, next_state, reward, episode_over])
        else:
            self.buffer.extend([state, action, next_state, reward, episode_over])
        if self.buffer.trainable and self.train:
            if self.step_count % self.train_freq == 0:
                self.update_model(self.buffer.sample(self.batchs))
            elif self.train_count % self.update_freq == 0:
                self.update_target()
            self.decay_epsilon()

    def update_target(self):
        if self.model:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            print('Model not created!')
            exit()

    def update_model(self, samples):
        self.train_count += 1
        self.model.eval()
        states = torch.Tensor([item[0] for item in samples]).to(self.device)
        next_states = torch.Tensor([item[2] for item in samples]).to(self.device)
        with torch.no_grad():
            current_qs = self.model(states)
            future_qs = self.target_model(next_states)

            for index, (_, action, _, reward, done) in enumerate(samples):
                if not done:
                    new_q = reward + self.y * torch.max(future_qs[index])
                else:
                    new_q = reward

                current_qs[index][action] = new_q

        self.model.train()

        preds = self.model(states)
        loss = self.loss_fn(preds, current_qs).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.train_count % 100 == 0:
            print(f"Train: {self.train_count} - loss ---> ", loss.item())
