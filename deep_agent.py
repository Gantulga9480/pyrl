import os
import torch
from .agent import Agent


class DeepAgent(Agent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size)
        self.device = device
        self.model = None
        self.optimizer = None
        self.loss_fn = None

    def create_model(self, model: torch.nn.Module, lr: float, gamma: float):
        self.lr = lr
        self.gamma = gamma
        self.model = model(self.state_space_size, self.action_space_size)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def save_model(self, path: str, as_jit: bool = False) -> None:
        if self.model and path:
            try:
                if as_jit:
                    model_scripted = torch.jit.script(self.model)
                    model_scripted.save(path)
                else:
                    torch.save(self.model.state_dict(), path)
            except Exception:
                os.makedirs("/".join(path.split("/")[:-1]))
                if as_jit:
                    model_scripted = torch.jit.script(self.model)
                    model_scripted.save(path)
                else:
                    torch.save(self.model.state_dict(), path)

    def load_model(self, path, from_jit: bool = False) -> None:
        if from_jit:
            self.model = torch.jit.load(path, map_location=self.device)
        else:
            self.model.load_state_dict(torch.load(path))
            self.model.to(self.device)
