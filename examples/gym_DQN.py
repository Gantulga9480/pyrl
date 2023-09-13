import torch
import torch.nn as nn
import numpy as np
import gym
import sys
import os
sys.path.append(os.getcwd())
from pyrl.dqn import DeepQNetworkAgent  # noqa
from pyrl.utils import ReplayBuffer     # noqa


class DQN(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.model(x)


torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
np.random.seed(3407)


ENV_NAME = "CartPole-v1"
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
env = gym.make(ENV_NAME, render_mode=None)
agent = DeepQNetworkAgent(env.observation_space.shape[0], env.action_space.n, device=DEVICE)
agent.create_model(DQN, lr=0.0001, gamma=0.99, e_decay=0.95, batch=64, target_update_method="soft", tau=0.01, tuf=10)
agent.create_buffer(ReplayBuffer(1_000_000, 1000, 4))
agent.e = 0.01

try:
    while agent.episode_counter < 1000:
        done = False
        s, i = env.reset(seed=3407)
        while not done:
            a = agent.policy(s)
            ns, r, d, t, i = env.step(a)
            done = d or t
            agent.learn(s, a, ns, r, done)
            s = ns
except KeyboardInterrupt:
    pass
env.close()

agent.eval()

env = gym.make(ENV_NAME, render_mode="human")
reward = 0
for _ in range(1):
    done = False
    state, _ = env.reset()
    while not done:
        action = agent.policy(state)
        state, r, d, t, _ = env.step(action)
        done = d or t
        reward += r
print(reward)
