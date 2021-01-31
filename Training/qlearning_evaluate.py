import numpy as np
import pickle
from qlearning_training import reset_enviroment, initialize_state_matrix, identifiesgoal_state, identifies_state

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

        next_state = identifies_state_matrix(i, j)
        reward = -1
        return reward, int(next_state)


def select_optimal_action(state):

    optimal = np.argmax(Q[state], axis=0)

    if Q[state][optimal] == 0.:
        optimal = np.argmin(Q[state], axis=0)
        print(state, optimal, Q[state][optimal])
    return optimal


def select_optimal_path(q_table, enviroment):
    enviroment[0][0] = 1
    # reset enviroment to learn a new goal
    i, j = identifies_state(enviroment, enviromentsize)
    k, l = identifiesgoal_state(enviroment, enviromentsize)
    state = int(state_matrix[i][j])
    goal_state = int(state_matrix[k][l])
    states = []
    states.append(state)
    print(state, goal_state, '\n')
    print('\n', q_table, '\n', enviroment)
    old_state, reward, next_state = 0, 0, 0
    done = False
    while(not done):
        action = select_optimal_action(state)
        reward, next_state = next_step(action, state, goal_state)
        state = next_state
        states.append(state)
        print(states)
        print(state, goal_state, '\n', enviroment, '\n', q_table)
        if reward == 10:
            done = True
    print(states, '\n', q_table)


def main():
    global state_matrix, enviromentsize, Q
    enviromentsize = 4
    env = np.zeros((enviromentsize, enviromentsize))
    env = reset_enviroment(env, enviromentsize)
    state_matrix = initialize_state_matrix(
        np.zeros((enviromentsize, enviromentsize)), enviromentsize)

    with open('pickle/Q_0_1.pickle', "rb") as read:
        Q = pickle.load(read)

    select_optimal_path(Q, env)


if __name__ == '__main__':
    main()
