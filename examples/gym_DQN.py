import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())
from PyRL.dqn import DeepQNetworkAgent  # noqa
from PyRL.utils import ReplayBuffer     # noqa


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
TRAIN_ID = "dqn_rewards_0.95"
env = gym.make(ENV_NAME, render_mode=None)
agent = DeepQNetworkAgent(env.observation_space.shape[0], env.action_space.n, device="cuda:0")
agent.create_model(DQN, lr=0.0001, y=0.99, e_decay=0.95, batch=64, target_update_method="soft", tau=0.01, tuf=10)
agent.create_buffer(ReplayBuffer(1_000_000, 1000, 4))
agent.e = 0.01

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

plt.xlabel(f"DQN - {TRAIN_ID}")
plt.plot(agent.reward_history)
plt.show()

with open(f"{TRAIN_ID}.txt", "w") as f:
    f.writelines([str(item) + '\n' for item in agent.reward_history])
