import numpy as np
import pickle
from qlearning_training import initialize_state_matrix, identifiesgoal_state, identifies_state
from matplotlib import pyplot as plt # pylint: disable=import-error
import plotly.express as px # pylint: disable=import-error
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
        print(state, optimal, Q[state][optimal])
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


def select_optimal_path(q_table, enviroment, state):
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
        states.append(state)
        if reward == 10:
            done = True
    states = states[:-1]
    define_steps()
    print(q_table, '\n', '\n', states, '\n', steps,'\n', len(steps), '\n', steps_desc)
    plot_q_with_steps(enviroment, states, enviromentsize, goal_state)
    steps_matrix.append(steps_desc)

def plot_q_with_steps(enviroment, steps, enviromentsize, state):
    for step in steps:
        if step!=state:
            i,j = identifies_index(step)
            enviroment[i][j] = 5
    plot_matrix(enviroment,enviromentsize,enviromentsize,state)

def plot_matrix(matrix, x_size, y_size, state):
    cmap = plt.cm.summer
    plt.figure(figsize=(10,10))
    plt.imshow(matrix, cmap=cmap)
    plt.xticks([x for x in range(x_size)])
    plt.yticks([y for y in range(y_size)])
    plt.title('Best Path to reach state '+str(state)+' in Enviroment')
    plt.tight_layout()
    plt.savefig("imgs/without_obstacles/Q/Q"+str(state)+"evaluate.png")


def reset_enviroment(enviroment, env_size, goal_position):
    enviroment = np.zeros((env_size, env_size))
    i, j = identifies_state_train(goal_position, env_size)
    enviroment[i][j] = 20
    enviroment[0][0] = 1
    return enviroment


def identifies_state_train(goal_position, size):
    for i in range(size):
        for j in range(size):
            if state_matrix[i][j] == goal_position:
                return i, j


def main():
    global state_matrix, enviromentsize, Q, steps_matrix
    steps_matrix = []
    enviromentsize = 10
    state_matrix = initialize_state_matrix(np.zeros((enviromentsize, enviromentsize)), enviromentsize)
    i=99
    env = np.zeros((enviromentsize, enviromentsize))
    env = reset_enviroment(env, enviromentsize, i)
    with open(r'C:\Users\mesqu\Downloads\TG\hexapo-robot-optmal\pickle\100.pickle', "rb") as read:
        Q = pickle.load(read)
        print(Q)
    select_optimal_path(Q, env, i)

    with open('pickle/without_obstacles/steps_positions.pickle', "wb") as write:
        pickle.dump(steps_matrix, write)

    print(steps_matrix)
    
if __name__ == '__main__':
    main()