#!/usr/bin/python
# -*-coding:utf-8-*-

import gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    # initialize empty dictionary of lists
    returns = defaultdict(list)
    # loop over episodes
    for i_episode in tqdm(range(episodes)):
        # generate an episode
        episode = generate_episodes(bj_env, strategy)
        # obtain the states, actions, and rewards
        states, actions, rewards = zip(*episode)
        # prepare for discounting
        discount = np.array([gamma ** i for i in range(len(rewards) + 1)])
        # calculate and store the return for each visit in the episode
        for i, state in enumerate(states):
            returns[state].append(sum(rewards[i:] * discount[:-(1 + i)]))
    # calculate the state-value function estimate
    V = {k: np.mean(v) for k, v, in returns.items()}
    return V


def mc_predict_q(bj_env, episodes, generate_episodes, strategy, gamma=1.0):
    # initialize empty dictionaries of arrays
    returns = defaultdict(lambda: np.zeros(bj_env.action_space.n))
    N = defaultdict(lambda: np.zeros(bj_env.action_space.n))
    Q = defaultdict(lambda: np.zeros(bj_env.action_space.n))
    # loop over episodes
    for i_episode in tqdm(range(episodes)):
        # generate an episode
        episode = generate_episodes(bj_env, strategy)
        # obtain the states, actions, and rewards
        states, actions, rewards = zip(*episode)
        # prepare for discounting
        discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
        # update the sum of the returns, number of visits
        for i, state in enumerate(states):
            returns[state][actions[i]] += sum(rewards[i:] * discounts[:-(1 + i)])
            N[state][actions[i]] += 1
        # function estimates for each state-action pair
        for state in returns.keys():
            Q[state] = returns[state] / N[state]
    return Q


def generate_episode_from_Q(bj_env, Q, epsilon, nA):
    """ generates an episode from following the epsilon-greedy policy """
    episode = []
    state = bj_env.reset()
    while True:
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
            if state in Q else bj_env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s


def update_Q_GLIE(episode, Q, N, gamma):
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]]
        old_N = N[state][actions[i]]
        Q[state][actions[i]] = old_Q + (sum(rewards[i:] * discounts[:-(1 + i)]) - old_Q) / (old_N + 1)
        N[state][actions[i]] += 1
    return Q, N


def update_Q_alpha(episode, Q, alpha, gamma):
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]]
        Q[state][actions[i]] = old_Q + alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)
    return Q


def mc_control_GLIE(bj_env, num_episodes, gamma=1.0):
    nA = env.action_space.n
    # initialize empty dictionaries of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    N = defaultdict(lambda: np.zeros(nA))
    # loop over episodes
    for i_episode in tqdm(range(0, num_episodes+1)):
        # set the value of epsilon
        epsilon = 1.0 / ((i_episode / 8000) + 1)
        # generate an episode by following epsilon-greedy policy
        episode = generate_episode_from_Q(bj_env, Q, epsilon, nA)
        # update the action-value function estimate using the episode
        Q, N = update_Q_GLIE(episode, Q, N, gamma)
    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k, np.argmax(v)) for k, v in Q.items())
    return policy, Q


def mc_control_alpha(env, num_episodes, alpha, gamma=1.0):
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    # loop over episodes
    for i_episode in tqdm(range(0, num_episodes + 1)):
        # set the value of epsilon
        epsilon = 1.0/((i_episode/8000)+1)
        # generate an episode by following epsilon-greedy policy
        episode = generate_episode_from_Q(env, Q, epsilon, nA)
        # update the action-value function estimate using the episode
        Q = update_Q_alpha(episode, Q, alpha, gamma)
    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k,np.argmax(v)) for k, v in Q.items())
    return policy, Q


def plot_value_function(V):
    def get_Z(x, y, usable_ace):
        if (x, y, usable_ace) in V:
            return V[x, y, usable_ace]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)

        Z = np.array([get_Z(x, y, usable_ace) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()


def plot_policy(policy):
    def get_Z(x, y, usable_ace):
        if (x, y, usable_ace) in policy:
            return policy[x, y, usable_ace]
        else:
            return 1

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(10, 0, -1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x, y, usable_ace) for x in x_range] for y in y_range])
        surf = ax.imshow(Z, cmap=plt.get_cmap('Pastel2', 2), vmin=0, vmax=1, extent=[10.5, 21.5, 0.5, 10.5])
        plt.xticks(x_range)
        plt.yticks(y_range)
        plt.gca().invert_yaxis()
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.grid(color='w', linestyle='-', linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[0, 1], cax=cax)
        cbar.ax.set_yticklabels(['0 (STICK)', '1 (HIT)'])

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(122)
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()


if __name__ == '__main__':
    # V = mc_predict_v(env, 100000, generate_episode_with_strategy, set_strategy_with_limit)
    # plot_value_function(V)

    # Q = mc_predict_q(env, 100000, generate_episode_with_strategy, set_strategy_with_limit_stochastic)
    # V_q = dict(((k, (k[0]>18)*(np.dot([0.8, 0.2], v)) + (k[0]<=18)*np.dot([0.2, 0.8], v))) for k, v in Q.items())
    # plot_value_function(V_q)

    # policy_glie, Q_glie = mc_control_GLIE(env, 100000)
    # V_glie = dict((k, np.max(v)) for k, v in Q_glie.items())
    # plot_value_function(V_glie)
    # plot_policy(policy_glie)

    policy_alpha, Q_alpha = mc_control_alpha(env, 100000, 0.02)
    V_alpha = dict((k, np.max(v)) for k, v in Q_alpha.items())
    plot_value_function(V_alpha)
    plot_policy(policy_alpha)
