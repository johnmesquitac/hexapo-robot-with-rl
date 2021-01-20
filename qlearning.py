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

def training_agent(q_table, enviroment, num_episodes):
    for i in range(num_episodes):
        enviroment = initialenviroment #reset enviroment to it's initial position

def plot_matrix(matrix, x_size, y_size):
    im = ax.matshow(matrix)
    for i in range(x_size):
        for j in range(y_size):
            text = ax.text(j, i, matrix[i, j],
                       ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show()

def main():
    global initialenviroment
    enviromentsize = 4
    state_size = enviromentsize*enviromentsize #defines environment
    env = np.zeros((enviromentsize, enviromentsize))
    env[0][0] = 1 #initial position
    env[3][3] = 2 #goal position
    initialenviroment = env #mantains a copy of the initialenvironment
    actions_size = enviromentsize #lef, right, up and down
    Q = np.zeros((state_size, actions_size)) #initializing q-table with zeros
    Q = training_agent(Q, env, 100)
if __name__ == '__main__':
    main()