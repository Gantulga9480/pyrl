import os
import torch
from .agent import Agent


class DeepAgent(Agent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size)
        self.model = None
        self.device = device
        self.optimizer = None
        self.loss_fn = None

    def create_model(self, model: torch.nn.Module, lr: float, y: float):
        self.lr = lr
        self.y = y
        self.model = model(self.state_space_size, self.action_space_size)
        self.model.to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def save_model(self, path: str) -> None:
        if self.model and path:
            try:
                torch.save(self.model.state_dict(), path)
            except Exception:
                os.makedirs("/".join(path.split("/")[:-1]))
                torch.save(self.model.state_dict(), path)

    def load_model(self, path) -> None:
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.model.train()
