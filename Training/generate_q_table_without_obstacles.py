import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import random
import pickle
from time import time
from matplotlib import pyplot as plt


# parameters of learning
alpha = 0.01  # learning rate, learn more quickly if alpha is closer to 1
gamma = 0.6  # use a higher gamma for smaller spaces because we value later rewards rather than former rewards
epsilon = 0.05  # agent can do 100% exploitation (0) or 100% exploration (1)

action_space = [0, 1, 2, 3]

NUM_EPISODES = 2000


'''action space:
    0 - south (up)
    1 - north(down)
    2 - east (right)
    3 - west (left)'''


def select_optimal_action(state, actions_allowed):
    #print('cheguei no seleciona melhor', state)
    for i in range(0, 3):
        if i in actions_allowed:
            pass
        else:
            Q[state][i] = 0
    # print(Q[state])
    optimal = np.argmax(Q[state], axis=0)
    if Q[state][optimal] == 0.:
        optimal = np.argmin(Q[state], axis=0)
    #print('select optimals', optimal)
    return optimal


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
    elif state in traps:
        return -5, state
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


def checking_allowed_spaces(state):
    if (state == 5 or state == 6 or state == 9 or state == 10):
        return np.array([0, 1, 2, 3])
    elif state == 0:
        return np.array([1, 2])
    elif (state == 1 or state == 2):
        return np.array([1, 2, 3])
    elif state == 3:
        return np.array([1, 3])
    elif state == 4 or state == 8:
        return np.array([0, 1, 2])
    elif state == 7 or state == 11:
        return np.array([0, 1, 3])
    elif state == 12:
        return np.array([0, 2])
    elif state == 13 or state == 14:
        return np.array([0, 2, 3])
    elif state == 15:
        return np.array([0, 3])


def update(enviroment, state, old_state, episode, steps):
    # print('updating')
    done = False
    #action_space = checking_allowed_spaces(state)
    if random.uniform(0, 1) < epsilon:
        action = random.choice(action_space)
    else:
        action = select_optimal_action(state, action_space)
    # get the reward of choosing current action and the next state
    reward, next_state = next_step(action, state, goal_state)

    if reward == 10 or reward == -5:
        done = True
    # stores the old_state to not going back avoiding loopings in the same square
    old_state = state

    # get maximum q_value for action in next state
    next_max_state_q_value = np.max(Q[next_state])
    Q[state][action] += alpha * \
        (reward + gamma*next_max_state_q_value - Q[state][action])

    return next_state, reward, old_state, done


def training_agent(enviroment, num_episodes, env_size, goal_position):
    global goal_state, steps, rewards, training_start, training_end, penalties
    training_start = time()
    steps = []
    rewards = []
    penalties = []
    states = []
    for episode in range(num_episodes):
        states = []
        enviroment = reset_enviroment(enviroment, env_size, goal_position)
        # reset enviroment to learn a new goal
        i, j = identifies_state(enviroment, env_size)
        k, l = identifiesgoal_state(enviroment, env_size)
        identifies_trap(enviroment, env_size)
        state = int(state_matrix[i][j])
        goal_state = int(state_matrix[k][l])
        states.append(state)
        epochs = 0
        num_penalties, reward, total_reward, old_state, penalty = 0, 0, 0, 0, 0
        done = False
        while not done:
            state, reward, old_state, done = update(
                enviroment, state, old_state, num_episodes, epochs)
            total_reward += reward
            epochs += 1
            if state in traps:
                penalty += 1
                num_penalties += penalty
            #print(state, reward, old_state, done, goal_state)
            states.append(state)
        # store steps and rewards for each episode
        steps.append(epochs)
        rewards.append(total_reward)
        penalties.append(num_penalties)

        print("Time steps: {}, Penalties: {}, Reward: {}, Goal State: {}, Epsilon:{}, Episode:{}".format(
            epochs, num_penalties, total_reward, goal_state, epsilon, episode))
        print(states)

    training_end = time()

    # print(rewards)


def evaluate_training(state):
    print(rewards)

    mean_rate = [np.mean(rewards[n-10:n]) if n > 10 else np.mean(rewards[:n])
                 for n in range(1, len(rewards))]
    elapsed_training_time = int(training_end-training_start)
    success_rate = np.mean(rewards)
    penalties_rate = np.mean(penalties)
    epochs_step_rate = np.mean(steps)

    print("\nThis environment has been solved", str(success_rate), "% of times over",  str(NUM_EPISODES), "episodes within", str(elapsed_training_time),
          "seconds and with an average number of penalties per episode", str(penalties_rate), "and an average number of timesteps per trip of", str(epochs_step_rate))

    plt.figure()
    plt.plot(rewards)
    plt.plot(mean_rate)
    plt.title('Gridworld Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig("imgs/without_obstacles/mean/mean"+state+".png")

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(rewards, '-g', label='reward')
    ax2 = ax1.twinx()
    ax2.plot(steps, '+r', label='step')
    ax1.set_xlabel("episode")
    ax1.set_ylabel("reward")
    ax2.set_ylabel("step")
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    plt.title("Training Progress")
    plt.savefig(
        'imgs/without_obstacles/trainingprocess/trainingprocess'+state+'.png')

    fig = plt.figure()
    plt.plot(steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.savefig('imgs/without_obstacles/steps/steps'+state+'.png')

    fig = plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.savefig('imgs/without_obstacles/rewards/rewards'+state+'.png')


# changes the goal to another place
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


def initialize_state_matrix(matrix, matrix_size):
    cont = 0
    for i in range(matrix_size):
        for j in range(matrix_size):
            matrix[i][j] = int(cont)
            cont += 1
    return matrix


def identifies_state(enviroment, env_size):
    for i in range(env_size):
        for j in range(env_size):
            if enviroment[i][j] == 1:
                return i, j


def identifiesgoal_state(enviroment, env_size):
    for i in range(env_size):
        for j in range(env_size):
            if enviroment[i][j] == 20:
                return i, j


def identifies_trap(enviroment, env_size):
    global traps
    traps = []
    for i in range(env_size):
        for j in range(env_size):
            if enviroment[i][j] == -1:
                traps.append(state_matrix[i][j])


# plot desired matrix
def plot_matrix(matrix, x_size, y_size, state):
    figure, ax = plt.subplots()
    im = ax.matshow(matrix)
    for i in range(x_size):
        for j in range(y_size):
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color="w")
    figure.tight_layout()
    plt.savefig("imgs/without_obstacles/Q/Q"+state+".png")


def main():
    global state_matrix, enviromentsize, Q
    enviromentsize = 4
    state_size = enviromentsize*enviromentsize  # defines environment
    for i in range(1, 16):
        state_matrix = initialize_state_matrix(
            np.zeros((enviromentsize, enviromentsize)), enviromentsize)
        env = np.zeros((enviromentsize, enviromentsize))
        env = reset_enviroment(env, enviromentsize, i)
        actions_size = enviromentsize  # lef, right, up and down
        # initializing q-table with zeros
        Q = np.zeros((state_size, actions_size))
        Q.shape
        training_agent(env, NUM_EPISODES, enviromentsize, i)
        evaluate_training(str(i))
        plot_matrix(env, enviromentsize, enviromentsize, str(i))
        print(Q)
        with open('pickle/'+str(i)+'.pickle', "wb") as write:
            pickle.dump(Q, write)


if __name__ == '__main__':
    main()
