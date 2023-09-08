import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())
from PyRL.one_step_actor_critic import OneStepActorCriticAgent as OSAC  # noqa


class Actor(nn.Module):

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


class Critic(nn.Module):

    def __init__(self, observation_size) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
np.random.seed(3407)


ENV_NAME = "CartPole-v1"
TRAIN_ID = "OSAC_rewards_itr5"
env = gym.make(ENV_NAME, render_mode=None)
agent = OSAC(env.observation_space.shape[0], env.action_space.n, device="cuda:0")
agent.create_model(Actor, Critic, actor_lr=0.001, critic_lr=0.001, entropy_coef=0.5, y=0.99)

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

plt.xlabel(f"{ENV_NAME} - {TRAIN_ID}")
plt.plot(agent.reward_history)
plt.show()

# with open(f"results/{TRAIN_ID}.txt", "w") as f:
#     f.writelines([str(item) + '\n' for item in agent.reward_history])
