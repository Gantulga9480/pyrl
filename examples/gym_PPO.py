import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())
from pyrl.ppo import ProximalPolicyOptimizationAgent as PPO  # noqa


class Actor(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(observation_size),
            nn.Linear(observation_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):

    def __init__(self, observation_size) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(observation_size),
            nn.Linear(observation_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
np.random.seed(3407)


ENV_NAME = "CartPole-v1"
TRAIN_ID = "ppo_rewards_norm_loss_mean"
env = gym.make(ENV_NAME, render_mode=None)
agent = PPO(env.observation_space.shape[0], env.action_space.n, device="cuda:0")
agent.create_model(Actor,
                   Critic,
                   actor_lr=0.0003,
                   critic_lr=0.003,
                   gamma=0.99,
                   gae_lambda=0.95,
                   entropy_coef=0.01,
                   vf_coef=1,
                   clip_coef=0.2,
                   target_kl=0.02,
                   max_grad_norm=1,
                   step_count=500,
                   batch=500,
                   epoch=100,
                   reward_norm_factor=1)

try:
    while agent.episode_counter < 1000:
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.policy(state)
            next_state, reward, d, t, _ = env.step(action)
            done = d or t
            agent.learn(state, action, next_state, reward, done)
            state = next_state
except KeyboardInterrupt:
    pass
env.close()

plt.xlabel(f"{ENV_NAME} - {TRAIN_ID}")
plt.plot(agent.reward_history)
plt.show()

# with open(f"results/{TRAIN_ID}.txt", "w") as f:
#     f.writelines([str(item) + '\n' for item in agent.reward_history])
# agent.training = False

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
