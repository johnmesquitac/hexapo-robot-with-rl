import numpy as np
import pickle
from qlearning_training import initialize_state_matrix, identifiesgoal_state, identifies_state
from matplotlib import pyplot as plt # pylint: disable=import-error
import plotly.express as px # pylint: disable=import-error
import pandas as pd 

action_space = np.array([0, 1, 2, 3])


def identifies_index(state):
    for i in range(enviromentsize):
        for j in range(enviromentsize):
            if state_matrix[i][j] == state:
                return i, j


def identifies_state_matrix(i, j):
    return state_matrix[i][j]
# verificar se tomando uma nova decisão não tomei a mesma anteriormente pra evitar caminhos longos


def next_step(action, state, goal_state):

    if state == goal_state:
        return 10, state
    else:
        i, j = identifies_index(state)

        # up
        if action == 0 and i > 0:
            i -= 1
        # Move left
        elif action == 1 and j > 0:
            j -= 1
        # Move down
        elif action == 2 and i < enviromentsize - 1:
            i += 1
        # Move right
        elif action == 3 and j < enviromentsize - 1:
            j += 1

        steps.append(action)
        next_state = identifies_state_matrix(i, j)
        reward = -1
        return reward, int(next_state)


def select_optimal_action(state):
    optimal = np.argmax(Q[state], axis=0)
    if Q[state][optimal] == 0:
        optimal = np.argmin(Q[state], axis=0)
    elif optimal == 0 and state == 0:
        optimal = np.argmin(Q[state], axis=0)
    return optimal


def define_steps():
    for step in steps:
        if step == 0:
            steps_desc.append('U')
        elif step == 1:
            steps_desc.append('L')
        elif step == 2:
            steps_desc.append('D')
        elif step == 3:
            steps_desc.append('R')


def select_optimal_path(q_table, enviroment, state_evaluate):
    global steps, steps_desc
    i, j = identifies_state(enviroment, enviromentsize)
    k, l = identifiesgoal_state(enviroment, enviromentsize)
    state = int(state_matrix[i][j])
    goal_state = int(state_matrix[k][l])
    states = []
    steps = []
    steps_desc = []
    states.append(state)
    reward, next_state = 0, 0
    done = False
    while(not done):
        action = select_optimal_action(state)
        reward, next_state = next_step(action, state, goal_state)
        state = next_state
        print(state)
        states.append(state)
        if reward == 10:
            done = True
    states = states[:-1]
    define_steps()
    print(q_table, '\n', '\n', states, '\n', steps, '\n', steps_desc)
    plot_q_with_steps(enviroment, states,enviromentsize, state_evaluate)
    steps_matrix.append(steps_desc)

def plot_q_with_steps(enviroment, steps, enviromentsize, state):
    for step in steps:
        if step!=state:
            i,j = identifies_index(step)
            enviroment[i][j] = 5
    plot_matrix(enviroment,enviromentsize,enviromentsize,state)


def reset_enviroment(enviroment, env_size, goal_position, obstacles_position):
    enviroment = np.zeros((env_size, env_size))
    i, j = identifies_state_train(goal_position, env_size)
    enviroment[i][j] = 20    
    for obstacle in obstacles_position:
        i,j = identifies_state_train(obstacle, env_size)
        enviroment[i][j] = -1
    enviroment[0][0] = 1
    return enviroment


def identifies_state_train(goal_position, size):
    for i in range(size):
        for j in range(size):
            if state_matrix[i][j] == goal_position:
                return i, j

def plot_matrix(matrix, x_size, y_size, state):
    cmap = plt.cm.gray
    plt.imshow(matrix, cmap=cmap)
    plt.xticks([x for x in range(x_size)])
    plt.yticks([y for y in range(y_size)])
    plt.title('Best Path to reach state '+str(state)+' in Enviroment')
    plt.tight_layout()
    plt.savefig("imgs/with_obstacles/Q/Q"+str(state)+"evaluate.png")

def main():
    global state_matrix, enviromentsize, Q, steps_matrix
    steps_matrix = []
    enviromentsize = 20
    state_matrix = initialize_state_matrix(
        np.zeros((enviromentsize, enviromentsize)), enviromentsize)
    obstacles = [20,21,23,24,26,27,29,30,32,33,35,36,38,39,40,41,43,44,46,47,49,50,52,53,55,56,58,59,60,61,63,64,66,67,69,70,72,73,75,76,78,79,80,81,83,84,86,87,89,90,92,93,95,96,98,99,
    120,121,123,124,126,127,129,130,132,133,135,136,138,139,140,141,143,144,146,147,149,150,152,153,155,156,158,159,160,161,163,164,166,167,169,170,172,173,175,176,178,179,180,181,183,184,
    186,187,189,190,192,193,195,196,198,199,220,221,223,224,226,227,229,230,232,233,235,236,238,239,240,241,243,244,246,247,249,250,252,253,255,256,258,259,260,261,263,264,266,267,269,270,
    272,273,275,276,278,279,280,281,283,284,286,287,289,290,292,293,295,296,298,299,320,321,323,324,326,327,329,330,332,333,335,336,338,340,341,343,344,346,347,349,350,352,353,355,356,
    358,360,361,363,364,366,367,369,370,372,373,375,376,378,380,381,383,384,386,387,389,390,392,393,395,396,398]
    i = 399
    state_matrix = initialize_state_matrix(
                np.zeros((enviromentsize, enviromentsize)), enviromentsize)
    env = np.zeros((enviromentsize, enviromentsize))
    env = reset_enviroment(env, enviromentsize, i, obstacles)
    with open('pickle/with_obstacles/'+str(i)+'.pickle', "rb") as read:
        Q = pickle.load(read)
    print('evaluating optimal path to position:',i)
    select_optimal_path(Q, env, i)

    print('steps', steps_matrix)

    with open('pickle/with_obstacles/steps_positions.pickle', "wb") as write:
        pickle.dump(steps_matrix, write)


if __name__ == '__main__':
    main()
