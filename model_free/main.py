'''
Yaeesh Mukadam
EECS 658 Assignment 8
Description: implementing RL monte carlo and epsilon greedy algorithms
Date: November 26, 2024
Inputs: none
Output: optimal gridworld policy
Collaborators: Lecture slides and chatgpt for debugging
'''

import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# importing numpy for working with arrays
# importing deque for a fixed-size error tracking window
# importing matplotlib for plotting graphs

class GridWorld:
    def __init__(self):
        self.grid = np.zeros((5, 5))
        self.terminal_states = [(0, 0), (4, 4)]
        self.actions = ['up', 'down', 'left', 'right']
        self.gamma = 0.9
    # initializes gridworld environment, terminal states, actions, and discount factor

    def step(self, state, action):
        x, y = state
        if state in self.terminal_states:
            return state, 0, True
        # immediately returns if the current state is terminal

        if action == 'up':
            next_state = (max(0, x - 1), y)
        elif action == 'down':
            next_state = (min(4, x + 1), y)
        elif action == 'left':
            next_state = (x, max(0, y - 1))
        elif action == 'right':
            next_state = (x, min(4, y + 1))
        else:
            next_state = state
        # calculates next state based on action while keeping it within grid boundaries

        reward = -1
        return next_state, reward, next_state in self.terminal_states
        # returns next state, default reward (-1), and whether the new state is terminal


def monte_carlo_first_visit(gw, threshold=1e-3):
    # function to implement first-visit monte carlo algorithm
    episodes = 0
    error_window_len = 3
    n = np.zeros((5, 5))  # count of visits to each state
    s = np.zeros((5, 5))  # cumulative rewards for each state
    v = np.zeros((5, 5))  # value function initialized to zeros
    prev_v = np.copy(v)  # for tracking changes between iterations
    errors = []  # stores the maximum error at each episode
    error_window = deque(maxlen=error_window_len)  # sliding window for error tracking
    threshold_flag = True  # flag to check if value function has converged

    print('Epoch 0:\n')
    print(f'inital values of N(s):\n{n}\n')
    print(f'inital values of S(s):\n{s}\n')
    print(f'inital values of V(s):\n{v}\n')
    # prints initial debug values

    while threshold_flag:
        state = (np.random.randint(0, 5), np.random.randint(0, 5))  # random start state
        done = False
        path = []  # stores the episode's state-action-reward triplets
        path_data = []  # stores debug info for printing
        g = []  # for calculating return

        while not done:
            action = np.random.choice(gw.actions)
            next_state, reward, done = gw.step(state, action)
            path.append((state, action, reward))
            state = next_state
        # simulates an episode by picking random actions

        visited = set()  # track first-visited states
        g = 0  # initialize return
        for k, elem in enumerate(reversed(path), start=1):
            state, action, reward = elem
            g = reward + gw.gamma * g
            x, y = state
            state_num = x * 5 + y + 1  # map state to a number for debugging
            path_data.append([k, state_num, reward, gw.gamma, g])
            if state not in visited:
                visited.add(state)  # only update first-visit states
                n[x, y] += 1
                s[x, y] += g
                v[x, y] = s[x, y] / n[x, y]
        # updates state values only for the first visit in the episode

        max_error = np.max(np.abs(v - prev_v))
        errors.append(max_error)
        error_window.append(max_error)
        prev_v = np.copy(v)  # updates previous values

        if len(error_window) == error_window_len and all(err < threshold for err in error_window):
            threshold_flag = False
        # stops iterating when error is consistently below threshold

        episodes += 1
        if episodes in [1, 10]:
            print(f'Epoch {episodes}:\n')
            print(f'Episode data array: \n{path_data}\n')
            print(f'N(s): \n{n}\n')
            print(f'S(s): \n{s}\n')
            print(f'V(s): \n{v}\n')
        # prints intermediate debug values at specific episodes

    print(f"Converged after {episodes} episodes.")
    print(f'Epoch {episodes}:\n')
    print(f'Episode data array: \n{path_data}\n')
    print(f'N(s): \n{n}\n')
    print(f'S(s): \n{s}\n')
    print(f'V(s): \n{v}\n')

    plt.figure(figsize=(8, 5))
    plt.axhline(y=1e-3, color='r', linestyle='--', label='y = 1e-3')
    plt.plot(errors, label='Max Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Monte Carlo First Visit')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.show()
    # plots the error convergence


def monte_carlo_every_visit(gw, threshold=1e-3):
    # function to implement every-visit monte carlo algorithm
    episodes = 0
    error_window_len = 3
    n = np.zeros((5, 5))  # count of visits to each state
    s = np.zeros((5, 5))  # cumulative rewards for each state
    v = np.zeros((5, 5))  # value function initialized to zeros
    prev_v = np.copy(v)  # for tracking changes between iterations
    errors = []  # stores the maximum error at each episode
    error_window = deque(maxlen=error_window_len)  # sliding window for error tracking
    threshold_flag = True  # flag to check if value function has converged

    print('Epoch 0:\n')
    print(f'inital values of N(s):\n{n}\n')
    print(f'inital values of S(s):\n{s}\n')
    print(f'inital values of V(s):\n{v}\n')
    # prints initial debug values

    while threshold_flag:
        state = (np.random.randint(0, 5), np.random.randint(0, 5))  # random start state
        done = False
        path = []  # stores the episode's state-action-reward triplets
        path_data = []  # stores debug info for printing
        g = []  # for calculating return

        while not done:
            action = np.random.choice(gw.actions)
            next_state, reward, done = gw.step(state, action)
            path.append((state, action, reward))
            state = next_state
        # simulates an episode by picking random actions

        g = 0  # initialize return
        for k, elem in enumerate(reversed(path), start=1):
            state, action, reward = elem
            g = reward + gw.gamma * g
            x, y = state
            state_num = x * 5 + y + 1  # map state to a number for debugging
            path_data.append([k, state_num, reward, gw.gamma, g])
            n[x, y] += 1
            s[x, y] += g
            v[x, y] = s[x, y] / n[x, y]
        # updates state values for every visit in the episode

        max_error = np.max(np.abs(v - prev_v))
        errors.append(max_error)
        error_window.append(max_error)
        prev_v = np.copy(v)  # updates previous values

        if len(error_window) == error_window_len and all(err < threshold for err in error_window):
            threshold_flag = False
        # stops iterating when error is consistently below threshold

        episodes += 1
        if episodes in [1, 10]:
            print(f'Epoch {episodes}:\n')
            print(f'Episode data array: \n{path_data}\n')
            print(f'N(s): \n{n}\n')
            print(f'S(s): \n{s}\n')
            print(f'V(s): \n{v}\n')
        # prints intermediate debug values at specific episodes

    print(f"Converged after {episodes} episodes.")
    print(f'Epoch {episodes}:\n')
    print(f'Episode data array: \n{path_data}\n')
    print(f'N(s): \n{n}\n')
    print(f'S(s): \n{s}\n')
    print(f'V(s): \n{v}\n')

    plt.figure(figsize=(8, 5))
    plt.axhline(y=1e-3, color='r', linestyle='--', label='y = 1e-3')
    plt.plot(errors, label='Max Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Monte Carlo Every Visit')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.show()
    # plots the error convergence


def monte_carlo_on_policy(gw, threshold=1e-3):
    # function to implement on-policy monte carlo algorithm
    episodes = 0
    error_window_len = 3
    n = np.zeros((5, 5))  # count of visits to each state
    s = np.zeros((5, 5))  # cumulative rewards for each state
    v = np.zeros((5, 5))  # value function initialized to zeros
    prev_v = np.copy(v)  # for tracking changes between iterations
    errors = []  # stores the maximum error at each episode
    error_window = deque(maxlen=error_window_len)  # sliding window for error tracking
    threshold_flag = True  # flag to check if value function has converged

    print('Epoch 0:\n')
    print(f'inital values of N(s):\n{n}\n')
    print(f'inital values of S(s):\n{s}\n')
    print(f'inital values of V(s):\n{v}\n')
    # prints initial debug values

    visited_initial = set()  # tracks which states have been visited
    all_visited = False  # flag for when all states have been visited

    while threshold_flag:
        state = (np.random.randint(0, 5), np.random.randint(0, 5))
        if len(visited_initial) < 23:
            while state in visited_initial:
                state = (np.random.randint(0, 5), np.random.randint(0, 5))
        # ensures exploration of unvisited states early on if fewer than 23 states have been visited

        done = False
        path = []  # stores the episode's state-action-reward triplets
        path_data = []  # stores debug info for printing
        g = []  # for calculating return
        steps = 0  # step counter to prevent infinite loops

        while not done and steps < 100:
            steps += 1
            x, y = state
            if state in visited_initial:
                action_values = {}
                for action in gw.actions:
                    next_state, _, _ = gw.step(state, action)
                    x, y = next_state
                    action_values[action] = v[x, y]
                best_action = max(action_values, key=action_values.get)
                next_state, reward, done = gw.step(state, best_action)
            else:
                action = np.random.choice(gw.actions)
                next_state, reward, done = gw.step(state, action)
            # picks the best action for visited states based on current value estimates
            # chooses a random action for unvisited states

            path.append((state, action, reward))
            state = next_state
        # simulates an episode using the current policy

        for elem in path:
            visited_initial.add(elem[0])
        # updates visited states

        g = 0  # initialize return
        for k, elem in enumerate(reversed(path), start=1):
            state, action, reward = elem
            g = reward + gw.gamma * g
            x, y = state
            state_num = x * 5 + y + 1  # map state to a number for debugging
            path_data.append([k, state_num, reward, gw.gamma, g])
            n[x, y] += 1
            s[x, y] += g
            v[x, y] = s[x, y] / n[x, y]
        # updates state values for every visit in the episode

        max_error = np.max(np.abs(v - prev_v))
        errors.append(max_error)
        error_window.append(max_error)
        prev_v = np.copy(v)  # updates previous values

        if len(error_window) == error_window_len and all(err < threshold for err in error_window):
            threshold_flag = False
        # stops iterating when error is consistently below threshold

        episodes += 1
        if episodes in [1, 10]:
            print(f'Epoch {episodes}:\n')
            print(f'Episode data array: \n{path_data}\n')
            print(f'N(s): \n{n}\n')
            print(f'S(s): \n{s}\n')
            print(f'V(s): \n{v}\n')
        # prints intermediate debug values at specific episodes

    print(f"Converged after {episodes} episodes.")
    print(f'Epoch {episodes}:\n')
    print(f'Episode data array: \n{path_data}\n')
    print(f'N(s): \n{n}\n')
    print(f'S(s): \n{s}\n')
    print(f'V(s): \n{v}\n')

    plt.figure(figsize=(8, 5))
    plt.axhline(y=1e-3, color='r', linestyle='--', label='y = 1e-3')
    plt.plot(errors, label='Max Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Monte Carlo On Policy Every Visit')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.show()
    # plots the error convergence


        

def main():
    gw = GridWorld()
    print('Part 1: Monte Carlo First Visit:\n')
    monte_carlo_first_visit(gw)
    print('\n\nPart 2: Monte Carlo Every Visit:\n')
    monte_carlo_every_visit(gw)
    print('\n\nPart 3: Monte Carlo On Policy Every Visit:\n')
    monte_carlo_on_policy(gw)
main()




        






