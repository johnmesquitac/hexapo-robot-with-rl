import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import random
from matplotlib import pyplot as plt

#plot parameters of
fig, ax = plt.subplots()

#parameters of learning
alpha = 0.1
gamma = 0.6
epsilon = 0.2

NUM_EPISODES = 100

'''action space:
    0 - south (up)
    1 - north(down)
    2 - east (right)
    3 - west (left)'''
action_space = np.array([0,1,2,3])

def select_optimal_action(q_table, state):
    return p.argmax(q_table[state],axis=0)
   

def next_step(q_table, enviroment, action, state):
    #check if next step is possible and give reward


def update(q_table, enviroment, state):
    if random.uniform(0,1) < epsilon:
        action = random.choice(action_space)
        print('action take', action)
    else:
        action = select_optimal_action(q_table, state)
    
    next_state =, reward = 

def reset_enviroment(enviroment, env_size): #changes the goal to another place
    enviroment = np.zeros((env_size, env_size))
    indices = np.random.randint(env_size, size=2)
    i, j = indices[0], indices[1]
    enviroment[i][j] = 20
    enviroment[0][0] = 1
    return enviroment


def training_agent(q_table, enviroment, num_episodes, env_size):
    for i in range(num_episodes):
        i,j = identifies_state(enviroment, env_size) #reset enviroment to learn a new goal
        state = int(state_matrix[i][j])
    epochs = 0
    num_penalties, reward, total_reward = 0, 0, 0
    while reward != 20:
        update(q_table, enviroment, state)
        total_reward += reward
        if reward == -10:
            num_penalties += 1
        epochs+=1

def initialize_state_matrix(matrix, matrix_size):
    cont = 0;
    for i in range(matrix_size):
        for j in range(matrix_size):
            matrix[i][j] = int(cont)
            cont+=1
    return matrix

def identifies_state(enviroment,env_size):
    env = reset_enviroment(enviroment, env_size)
    for i in range(env_size):
        for j in range(env_size):
            if env[i][j] == 1:
                return i,j

def plot_matrix(matrix, x_size, y_size):
    im = ax.matshow(matrix)
    for i in range(x_size):
        for j in range(y_size):
            text = ax.text(j, i, matrix[i, j],
                       ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show()

def main():
    global state_matrix
    enviromentsize = 4
    state_size = enviromentsize*enviromentsize #defines environment
    env = np.zeros((enviromentsize, enviromentsize))
    env = reset_enviroment(env, enviromentsize)
    actions_size = enviromentsize #lef, right, up and down
    state_matrix = initialize_state_matrix(np.zeros((enviromentsize, enviromentsize)), enviromentsize)
    Q = np.zeros((state_size, actions_size)) #initializing q-table with zeros
    Q = training_agent(Q, env, NUM_EPISODES, enviromentsize)

if __name__ == '__main__':
    main()