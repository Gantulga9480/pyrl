import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import sys
import os
sys.path.append(os.getcwd())
from PyRL import DeepDeterministicPolicyGradientAgent as DDPGAgent  # noqa
from PyRL.utils import ReplayBuffer                                 # noqa


class Actor(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, 16),
            nn.LeakyReLU(),
            nn.Linear(16, action_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size + action_size, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))


torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
np.random.seed(3407)


ENV_NAME = "MountainCarContinuous-v0"
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
env = gym.make(ENV_NAME, render_mode=None)
agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0], device=DEVICE)
agent.create_model(Actor, Critic, actor_lr=0.003, critic_lr=0.003, gamma=0.99, noise_std=0.3, batch=64, tau=0.01)
agent.create_buffer(ReplayBuffer(1_000_000, 1000, env.observation_space.shape[0], env.action_space.shape[0]))

try:
    while agent.episode_counter < 1000:
        done = False
        s, info = env.reset(seed=3407)
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
