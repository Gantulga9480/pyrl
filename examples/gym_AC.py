import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())
from pyrl.actor_critic import ActorCriticAgent  # noqa


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
TRAIN_ID = "ac_rewards_norm_loss_mean_itr5"
ENV_COUNT = 1
envs = [gym.make(ENV_NAME, render_mode=None) for _ in range(ENV_COUNT)]
agent = ActorCriticAgent(envs[0].observation_space.shape[0], envs[0].action_space.n, device="cuda:0")
agent.create_model(Actor, Critic, actor_lr=0.001, critic_lr=0.001, gamma=0.99, entropy_coef=0.1, gae_lambda=1, env_count=ENV_COUNT, step_count=500, reward_norm_factor=1)

states = np.zeros((ENV_COUNT, envs[0].observation_space.shape[0]))
next_states = np.zeros((ENV_COUNT, envs[0].observation_space.shape[0]))
rewards = np.zeros(ENV_COUNT)
dones = np.zeros(ENV_COUNT)
try:
    while agent.episode_counter < 1000:
        for i, env in enumerate(envs):
            s, _ = env.reset()
            states[i] = s
            dones[i] = 0
        while not any(dones):
            actions = agent.policy(states)
            for i, env in enumerate(envs):
                rewards[i] = 0
                if not dones[i]:
                    ns, r, d, t, _ = env.step(actions[i])
                    next_states[i] = ns
                    rewards[i] = r
                    dones[i] = d or t
            agent.learn(states, actions, next_states, rewards, dones)
            states = np.copy(next_states)
except KeyboardInterrupt:
    pass
env.close()

plt.xlabel(f"{ENV_NAME} - {TRAIN_ID}")
plt.plot(agent.reward_history)
plt.show()

# with open(f"results/{TRAIN_ID}.txt", "w") as f:
#     f.writelines([str(item) + '\n' for item in agent.reward_history])
agent.training = False

env = gym.make(ENV_NAME, render_mode="human")
for _ in range(10):
    done = False
    state, _ = env.reset()
    while not done:
        action = agent.policy(state)[0]
        state, _, d, t, _ = env.step(action)
        done = d or t
