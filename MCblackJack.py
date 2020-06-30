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
    if score > n:
        return 1
    else:
        return 0


def set_strategy_with_limit_stochastic(score, n=18):
    if score > n:
        probs = [0.8, 0.2]
    else:
        probs = [0.2, 0.8]
    return np.random.choice(np.arange(2), p=probs)


def generate_episode_with_strategy(bj_env, strategy):
    episode = []
    state = bj_env.reset()
    while True:
        action = int(strategy(state[0]))
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def mc_predict_v(bj_env, episodes, generate_episodes, strategy, gamma=1.0):
    returns = defaultdict(list)
    for i_episode in tqdm(range(episodes)):
        episode = generate_episodes(bj_env, strategy)
        states, actions, rewards = zip(*episode)
        discount = np.array([gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            returns[state].append(sum(rewards[i:]*discount[:-(1+i)]))
    V = {k: np.mean(v) for k, v, in returns.items()}
    return V


def mc_predict_q(bj_env, episodes, generate_episodes, strategy, gamma=1.0):
    returns = defaultdict(lambda: np.zeros(bj_env.action_space.n))
    N = defaultdict(lambda: np.zeros(bj_env.action_space.n))
    Q = defaultdict(lambda: np.zeros(bj_env.action_space.n))

    for i_episode in tqdm(range(1, episodes+1)):
        episode = generate_episodes(bj_env, strategy)
        states, actions, rewards = zip(*episode)
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            returns[state][actions[i]] += sum(rewards[i:]*discounts[:-(1+i)])
            N[state][actions[i]] += 1
            Q[state][actions[i]] += returns[state][actions[i]] / N[state][actions[i]]
    return Q


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
    V = mc_predict_v(env, 50000, generate_episode_with_strategy, set_strategy_with_limit)
    plot_value_function(V)

    # Q = mc_predict_q(env, 50000, generate_episode_with_strategy, set_strategy_with_limit_stochastic)

    # V_to_plot = dict(((k, (k[0]>18)*(np.dot([0.8, 0.2], v)) + (k[0]<=18)*np.dot([0.2, 0.8], v))) for k, v in Q.items())
    # plot_value_function(V_to_plot)

