import numpy as np
import pickle
from qlearning_training import initialize_state_matrix, identifiesgoal_state, identifies_state
from matplotlib import pyplot as plt # pylint: disable=import-error

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
    figure, ax = plt.subplots()
    im = ax.matshow(matrix, cmap=cmap)
    #figure.tight_layout()
    plt.savefig("imgs/with_obstacles/Q/Q"+str(state)+"evaluate.png")

def main():
    global state_matrix, enviromentsize, Q, steps_matrix
    steps_matrix = []
    enviromentsize = 15
    state_matrix = initialize_state_matrix(
        np.zeros((enviromentsize, enviromentsize)), enviromentsize)
    obstacles = [32,33,34,38,39,40,41,42,43,48,49,50,54,55,56,57,58,64,65,66,70,71,72,73,79,80,81,85,86,87,88,95,96,98,101,102,103,106,107,109,110,111,112,113,116,117,121,122,124,125,126,127,132,136,137,139,140,141,142,147,156,157,161,162,166,167,171,172,173,177,181,182,186,187,188,192,193,196,197,203,206,207,208] 
    i = 224
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
