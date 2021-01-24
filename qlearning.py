import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import random
from time import time
from matplotlib import pyplot as plt


#parameters of learning
alpha = 0.01
gamma = 0.95
epsilon = 0.1

NUM_EPISODES = 1000

'''action space:
    0 - south (up)
    1 - north(down)
    2 - east (right)
    3 - west (left)'''
action_space = np.array([0,1,2,3])

def select_optimal_action(q_table, state):
    return np.argmax(q_table[state],axis=0)

def check_old_state(old_state, new_state):
    if new_state == old_state: 
        return -10, old_state
    else: 
        return -1, new_state

def next_step(q_table, enviroment, action, state, old_state): #verificar se tomando uma nova decisão não tomei a mesma anteriormente pra evitar caminhos longos
    #print('estou no estado', state, 'acao', action, 'vim do estado', old_state )
    if state == goal_state:
        return 20, state
    else:
        if state == 0 :
            if action == 0 or action==3:
                return -10, state
            elif action==1:
                return -1, state+4
            elif action==2:
                return  -1, state+1

        if state == 1 or state == 2:
            if action == 0 or action ==3: 
                return -10, state
            elif action==1:
                reward, new_state = check_old_state(old_state, state+4)
                return reward, new_state
            elif action==2:
                reward, new_state = check_old_state(old_state, state+1)
                return reward, new_state

        if state == 5 or state == 6 or state==9 or state==10:
            if action == 0:
                reward, new_state = check_old_state(old_state, state-4)
                return reward, new_state
            elif action == 1:
                reward, new_state = check_old_state(old_state, state+4)
                return reward, new_state
            elif action == 2:
                reward, new_state = check_old_state(old_state, state+1)
                return reward, new_state
            elif action == 3:
                reward, new_state = check_old_state(old_state, state-1)
                return reward, new_state
        
        if state == 3:
            if action == 0 or action==2:
                return -10, state
            elif action == 1:
                reward, new_state = check_old_state(old_state, state+4)
                return reward, new_state
            elif action == 3:
                reward, new_state = check_old_state(old_state, state-1)
                return reward, new_state

        if state == 7 or state == 11:
            if action == 0:
                reward, new_state = check_old_state(old_state, state-4)
                return reward, new_state
            elif action == 1:
                reward, new_state = check_old_state(old_state, state+4)
                return reward, new_state
            elif action == 2:
                return -10, state
            elif action == 3:
                reward, new_state = check_old_state(old_state, state-1)
                return reward, new_state

        if state == 4 or state ==8:
            if action == 0 :
                reward, new_state = check_old_state(old_state, state-4)
                return reward, new_state
            elif action==3:
                return -10, state
            elif action==1:
                reward, new_state = check_old_state(old_state, state+4)
                return reward, new_state
            elif action==2:
                reward, new_state = check_old_state(old_state, state+1)
                return reward, new_state
        if state == 12:
            if action == 0 :
                reward, new_state = check_old_state(old_state, state-4)
                return reward, new_state
            elif action==3 or action==1:
                return -10, state
            elif action==2:
                reward, new_state = check_old_state(old_state, state+1)
                return reward, new_state        
        if state == 13 or state == 14:
            if action == 0 :
                reward, new_state = check_old_state(old_state, state-4)
                return reward, new_state
            elif action==3: 
                reward, new_state = check_old_state(old_state, state-1)
                return reward, new_state
            elif action==1:
                return -10, state
            elif action==2:
                reward, new_state = check_old_state(old_state, state+1)
                return reward, new_state
        if state == 15:
            if action == 0 :
                reward, new_state = check_old_state(old_state, state-4)
                return reward, new_state
            elif action==3: 
                reward, new_state = check_old_state(old_state, state-1)
                return reward, new_state
            elif action==1 or action==2:
                return -10, state


def update(q_table, enviroment, state, old_state, episode):
    if random.uniform(0,1) < epsilon:
        action = random.choice(action_space)
    else:
        action = select_optimal_action(q_table, state)
    reward, next_state = next_step(q_table, enviroment, action, state, old_state)
    old_state = state
    old_q_value = q_table[state][action]
    #get maximum q_value for action in next state
    next_max_state_q_value = np.argmax(q_table[next_state],axis=0)
    if episode == 1000 and reward==20:
         new_q_value = (1-alpha) * old_q_value + alpha * (-1 + gamma * next_max_state_q_value)
    else:
        new_q_value = (1-alpha) * old_q_value + alpha * (reward + gamma * next_max_state_q_value)
    q_table[state][action] = new_q_value
    return next_state, reward, old_state


def training_agent(q_table, enviroment, num_episodes, env_size):
    global goal_state, steps, rewards, training_start, training_end
    training_start = time()
    steps = []
    rewards = []
    for x in range(num_episodes):
        i,j = identifies_state(enviroment, env_size) #reset enviroment to learn a new goal
        k, l = identifiesgoal_state(enviroment, env_size)
        state = int(state_matrix[i][j])
        goal_state = int(state_matrix[k][l])
        epochs = 0
        num_penalties, reward, total_reward, old_state = 0, 0, 0, 0
        while int(reward) < 20 :
            state,reward,old_state = update(q_table, enviroment, state, old_state, num_episodes)
            total_reward += reward
            epochs+=1
            if reward == -10:
                num_penalties+=1

        #store steps and rewards for each episode
        steps.append(epochs)
        rewards.append(total_reward)
        print("Time steps: {}, Penalties: {}, Reward: {}".format(epochs,num_penalties,total_reward))
        training_end = time()
    print(rewards)
    return q_table, enviroment

def evaluate_training():
    print(rewards)
    mean_rate = [np.mean(rewards[n-10:n]) if n > 10 else np.mean(rewards[:n]) 
               for n in range(1, len(rewards))]
    elapsed_training_time = int(training_end-training_start)
    success_rate = np.mean(rewards)
    print("\nThis environment has been solved", str(success_rate), "% of times over",  str(NUM_EPISODES), "episodes within", str(elapsed_training_time), "seconds!")
    plt.figure(figsize=(12,8))
    plt.plot(rewards)
    plt.plot(mean_rate)
    plt.title('Gridworld Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig("mean.png")
    
def reset_enviroment(enviroment, env_size): #changes the goal to another place
    enviroment = np.zeros((env_size, env_size))
    indices = np.random.randint(env_size, size=2)
    i, j = indices[0], indices[1]
    enviroment[i][j] = 20
    enviroment[0][0] = 1
    return enviroment

def initialize_state_matrix(matrix, matrix_size):
    cont = 0
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

def identifiesgoal_state(enviroment,env_size):
    for i in range(env_size):
        for j in range(env_size):
            if enviroment[i][j] == 20:
                return i,j

def plot_matrix(matrix, x_size, y_size):
    #plot parameters of
    figure, ax = plt.subplots() 
    im = ax.matshow(matrix)
    for i in range(x_size):
        for j in range(y_size):
            text = ax.text(j, i, matrix[i, j],
                       ha="center", va="center", color="w")
    figure.tight_layout()
    plt.savefig("Q_table.png")

def main():
    global state_matrix
    enviromentsize = 4
    state_size = enviromentsize*enviromentsize #defines environment
    env = np.zeros((enviromentsize, enviromentsize))
    env = reset_enviroment(env, enviromentsize)
    actions_size = enviromentsize #lef, right, up and down
    state_matrix = initialize_state_matrix(np.zeros((enviromentsize, enviromentsize)), enviromentsize)
    Q = np.zeros((state_size, actions_size)) #initializing q-table with zeros
    Q, enviroment = training_agent(Q, env, NUM_EPISODES, enviromentsize)
    evaluate_training()
    print(Q)



if __name__ == '__main__':
    main()