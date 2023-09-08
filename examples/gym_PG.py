import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())
from PyRL.reinforce import ReinforceAgent  # noqa


class PG(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
np.random.seed(3407)


ENV_NAME = "CartPole-v1"
TRAIN_ID = "ri_rewards_sum"
env = gym.make(ENV_NAME, render_mode=None)
agent = ReinforceAgent(env.observation_space.shape[0], env.action_space.n, device="cuda:0")
agent.create_model(PG, lr=0.0001, entropy_coef=0.7, y=0.99)

try:
    while agent.episode_count < 1000:
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

plt.xlabel(f"{ENV_NAME} - {TRAIN_ID}")
plt.plot(agent.reward_history)
plt.show()

# with open(f"{TRAIN_ID}.txt", "w") as f:
#     f.writelines([str(item) + '\n' for item in agent.reward_history])
