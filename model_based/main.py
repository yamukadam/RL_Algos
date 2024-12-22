'''
Yaeesh Mukadam
EECS 658 Assignment 7
Description: implementing RL policy iteration and value iteration algorithms
Date: November 26, 2024
Inputs: none
Output: optimal gridworld policy and value results
Collaborators: Lecture slides and chatgpt for debugging and 
https://medium.com/@ngao7/markov-decision-process-policy-iteration-42d35ee87c82#c6bc,
https://medium.com/@ngao7/markov-decision-process-value-iteration-2d161d50a6ff
for help
'''

import numpy as np
#import matplotlib.pyplot as plt

ROWS = 5
COLS = 5
#defining rows and cols for the gridworld

threshold = 1e-6
#setting threshold for the convergence

moves = [[1,0], [-1,0], [0,1], [0,-1]]
#possible moves

termination_states = [(0,0), (4,4)]
#the 2 termination states in top left and bottom right corners

def get_next_state(state, action):
    #func to get next state for each move, if move is out of bounds (hits wall) then will stay in same spot
    new_x = state[0] + action[0]
    new_y = state[1] + action[1]
    if 0 <= new_x <= 4 and 0 <= new_y <= 4:
        return (new_x, new_y)
    return state
    
def get_reward(state):
    #func defining the reward obtained from moving from spot
    if state in termination_states:
        return 0
    return -1


def initialize_policy():
    #initialzing grid world matrix
    policy = np.zeros((ROWS,COLS))
    return policy



def policy_evaluation(policy):
    #func to do policy iteration
    counter = 0
    #counter to store num of iterations
    threshold_flag = True
    #boolean threshold flag to mark whether or not algo has converged
    errors_to_plot = []
    #array to store errors to plot
    while threshold_flag:
        #while not converged
        counter+=1
        #increment counter
        prev_state = policy.copy()
        #store prev state for future use
        errors = []
        #array to store deltas for each state in matrix
        for r in range(ROWS):
            for c in range(COLS):
                #iterate through each element in matrix
                if (r,c) in termination_states:
                    #if state is already in termination then no need to do anything
                    continue
                state = [r,c]
                #store state
                prev_value = prev_state[r][c]
                #get curr value
                new_value = -1
                #reward for all transitions is -1
                for move in moves:
                    #iterate through all possible moves
                    next_state = get_next_state(state,move)
                    #do move
                    if next_state == state:
                        #if hit wall, then same val remains
                        val = prev_value * .25
                    else:
                        #else get new val
                        val = prev_state[next_state[0]][next_state[1]] * .25
                    #add the weighted averages for each move together
                    new_value += val
                policy[r][c] = new_value
                #update matrix
                errors.append(abs(new_value - prev_value))
                #append deltas
        if counter == 1:
            #print statements as per assignment requirements
            print(f'Iteration 1:\n{policy}')
        if counter == 10:
            print(f'Iteration 10:\n{policy}')
        max_error = max(errors)
        #get the max delta b/c we want to check if even one of the deltas is above the threshold then we keep going
        errors_to_plot.append(max_error)
        #append the max
        if max_error < threshold:
            #if max is below threshold than that means every single delta is below
            threshold_flag = False
            #so it has converged
    return policy,counter, errors_to_plot
    #return optimal matrix, num of iterations, and array of errors
def part1():
    #run the policy iteration algorithm and plot the error value vs num of iterations
    policy = initialize_policy()
    print(f'Iteration 0:\n{policy}')
    optimal_policy,count, errors_to_plot = policy_evaluation(policy)
    print(f'Final Iteration (Number: {count}):\n{policy}')
    plt.figure(figsize=(8, 5))
    plt.axhline(y=1e-6, color='r', linestyle='--', label='y = 1e-6')
    plt.plot(errors_to_plot, label='Max Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Policy iteration convergence')
    plt.grid(True)
    plt.legend()
    #plt.yscale('log')
    plt.show()


def value_iteration(values):
    #func to do value iteration algorithm
    counter = 0
    #counter to see how many iterations
    threshold_flag = True
    #boolean threshold flag to mark whether or not algo has converged
    errors_to_plot = []
    #array to hold deltas for each iteration to plot
    while threshold_flag:
        #while it has not converged
        counter +=1
        #increment counter
        old_values = values.copy()
        #store curr matrix
        errors = []
        #array to hold deltas for each state
        for r in range(ROWS):
            for c in range(COLS):
                #iterate through each state
                if (r,c) in termination_states:
                    #if state is already termination state then no need to do anything
                    continue
                vals = []
                #storing the value for each possible move
                cur_val = values[r][c]
                #value of state before curr iteration
                for move in moves:
                    #go through each possible move
                    new_state = get_next_state([r,c], move)
                    #get new state
                    reward = get_reward((r,c))
                    #get reward for moving from current state which should be -1
                    vals.append(old_values[new_state[0]][new_state[1]] + reward)
                    #append the value from each possible move to vals
                values[r][c] = max(vals)
                #make val at curr state to be the max of all the possible moves
                errors.append(abs(cur_val - values[r][c]))
                #add deltas for each state
        if counter == 1:
            #print statements as per assignment instructions
            print(f'Iteration {counter}:\n {values}')
        if counter == 2:
            print(f'Iteration {counter}:\n {values}')
        
        max_error = max(errors)
        #get the max delta from all the states
        errors_to_plot.append(max_error)
        #append the max to array storing errors for plotting

        if max_error <= threshold:
            #if any of the deltas are below threshold then convergence has happened
            threshold_flag = False
    return values, counter, errors_to_plot
    #return optimal matrix, num of iterations, and array of errors to plot
def part2():
    #func to run the value iteration algorithm and plot the error values vs num of iteration
    values = initialize_policy()
    print(f'Iteration 0:\n {values}')
    optimal_values,counter,errors_to_plot = value_iteration(values)
    print(f'Final Iteration (Number: {counter}):\n {optimal_values}')
    plt.figure(figsize=(5, 5))
    plt.axhline(y=1e-6, color='r', linestyle='--', label='y = 1e-6')
    plt.plot(errors_to_plot, label='Max Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Value iteration convergence')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    #run each function
    print("PART 1 - POLICY ITERATION: \n")
    part1()
    print("\n\n\nPART 2 - VALUE ITERATION: \n")
    part2()
main()