from q_learning_env import enviroment
from evaluate_training import evaluate_training
import numpy as np
import matplotlib
import random
import time
import math

alpha = 0.6  # learning rate, learn more quickly if alpha is closer to 1
gamma = 0.95  # use a higher gamma for bigger spaces because we value later rewards rather than former rewards
lambda_val = 500
epsilon_decay = 0.1
min_eps = 0.0005
NUM_EPISODES = 750

env = enviroment()
Q = np.zeros((env.states_size,env.actions_size)).tolist()
EM_table = np.zeros(env.states_size).tolist()
steps = []
rewards = []
penalties = []
states = []

def calculate_dynamic_reward(xold, yold, xnext, ynext, xend, yend):
    xf = pow((xend-xold),2)
    yf = pow((yend-yold),2)
    dt = math.sqrt((xf+yf))
    xfnext = pow((xend-xnext),2)
    yfnext = pow((yend-ynext),2)
    dtf = math.sqrt((xfnext+yfnext))
    subdtf = dt-dtf 
    if subdtf!=0:
        rewardend = lambda_val*((subdtf)/abs(subdtf))
        return rewardend
    else:
         return 0

def select_optimal_action(state):
    best = Q[state].index(max(Q[state]))
    return best

def take_next_step(env, done, state, epsilon):
    obstacle = False
    reward = 0
    old_x, old_y = env.get_state_index()
    env.insert_old_state_index(old_x, old_y)
    if np.random.random() < epsilon:
        action = env.select_random_action()
    else:
        action = select_optimal_action(state)
    next_state,reward, done, obstacle = env.next_step(action, state)
    next_x, next_y = env.get_state_index()
    goal_x, goal_y = env.get_goal_index()
    reward_dynamic = calculate_dynamic_reward(old_x, old_y, next_x, next_y, goal_x, goal_y)
    reward += reward_dynamic
    next_best_action = np.argmax(Q[next_state])
    next_max_state_q_value = Q[next_state][next_best_action]
    Q[state][action] += alpha * (reward + gamma*next_max_state_q_value - Q[state][action])
    return done, next_state, reward

def main():
    epsilon = 0.9 # agent can do 100% exploitation (0) or 100% exploration (1)
    steps = []
    rewards = []
    penalties = []
    states = []
    steps2  = 0
    training_start = time.time()
    for i in range(NUM_EPISODES):
        print('episode', i)
        state, reward, done = env.reset_enviroment()
        step, total_reward = 0, 0
        while not done:
            steps2=0
            done, state, reward = take_next_step(env, done, state, epsilon)
            step+=1
            steps2+=1
            total_reward+=reward
            states.append(state)
            env.update_instructions_list(steps2, states)
        epsilon -= (epsilon_decay*epsilon) if epsilon>min_eps else 0
        steps.append(step)
        rewards.append(total_reward)
    training_end = time.time()
    '''    print(Q[891])
        print(max(Q[891]))
        print(Q[891].index(max(Q[891])))
        print(Q[891][Q[891].index(max(Q[891]))])'''
    evaluate_training(str(env.states_size),steps, rewards, training_end, training_start, NUM_EPISODES, Q)

if __name__=='__main__':
    main()       
    
