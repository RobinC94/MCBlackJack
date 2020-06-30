#!/usr/bin/python
# -*-coding:utf-8-*-

import gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


env = gym.make("Blackjack-v0")


def set_strategy_with_limit(score, n=18):
    if score > 18:
        return 1
    else:
        return 0


def generate_episode_with_strategy(bj_env, set_strategy):
    episode = []
    state = bj_env.reset()
    while True:
        action = int(set_strategy(state[0]))
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def mc_predict_v(bj_env, episodes, generate_episodes, gamma=1.0):
    returns = defaultdict(list)
    for i_episode in tqdm(range(episodes)):
        episode = generate_episodes(env, set_strategy_with_limit(18))
        states, actions, rewards = zip(*episode)
        discount = np.array([gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            returns[state].append(sum(rewards[i:]*discount[:-(1+i)]))
    V = {k: np.mean(v) for k, v, in returns.items()}
    return V


def plot_value_function(Q_table):
    x = np.arange(12, 21)
    y = np.arange(1, 10)
    X, Y = np.meshgrid(x, y)
    Z_noace = np.apply_along_axis(lambda x: Q_table[(x[0], x[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda x: Q_table[(x[0], x[1], True)], 2, np.dstack([X, Y]))
    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', vmin=-1.0, vmax=1.0)
        ax.set_xlabel('\nPlayer Sum', fontsize=25)
        ax.set_ylabel('\nDealer Showing', fontsize=25)
        ax.set_zlabel('\nValue', fontsize=25)
        ax.set_title('\n\n' + title, fontsize=25)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)

    plot_surface(X, Y, Z_noace, 'optimal value function(not use Ace)')
    plot_surface(X, Y, Z_ace, 'optimal value function(use Ace)')
    plt.show()


if __name__ == '__main__':
    V = mc_predict_v(env, 500000, generate_episode_with_strategy)
    plot_value_function(V)


# for i in range(3):
#     state = env.reset()
#     while True:
#         print(state)
#         action = env.action_space.sample()
#         state, reward, done, info = env.step(action)
#         if done:
#             print("End game! Reward: ", reward)
#             print("You win!") if reward>0 else print("You lose! ")
#             break
