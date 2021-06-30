from q_learning_env import enviroment
from evaluate_training import evaluate_training
import numpy as np
import matplotlib
import random
import time
import math

alpha = 0.006  # learning rate, learn more quickly if alpha is closer to 1
gamma = 0.6  # use a higher gamma for smaller spaces because we value later rewards rather than former rewards
lambda_val = 100
epsilon_decay = 0.1
NUM_EPISODES = 150

env = enviroment()
Q = np.zeros((env.states_size,env.actions_size)).tolist()
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
    return Q[state].index(max(Q[state]))

def take_next_step(env, done, state, epsilon):
    old_x, old_y = env.get_state_index()
    if random.uniform(0,1) < epsilon:
        action = env.select_random_action()
    else:
        action = select_optimal_action(state)
    next_state,reward, done = env.next_step(action)
    next_x, next_y = env.get_state_index()
    goal_x, goal_y = env.get_goal_index()
    reward_dynamic = calculate_dynamic_reward(old_x, old_y, next_x, next_y, goal_x, goal_y)
    reward += reward_dynamic
    next_max_state_q_value = np.max(Q[next_state])
    Q[state][action] += alpha * (reward + gamma*next_max_state_q_value - Q[state][action])  
    return done, next_state, reward

def main():
    epsilon = 0.05  # agent can do 100% exploitation (0) or 100% exploration (1)
    steps = []
    rewards = []
    penalties = []
    states = []
    training_start = time.time()
    for _ in range(NUM_EPISODES):
        state, reward, done = env.reset_enviroment()
        step, total_reward = 0, 0
        while not done:
            done, state, reward = take_next_step(env, done, state, epsilon)
            step+=1
            total_reward+=reward
            states.append(state)
        #epsilon -= (epsilon_decay*epsilon)
        steps.append(step)
        rewards.append(total_reward)
    training_end = time.time()
    evaluate_training(str(9999),steps, rewards, training_end, training_start, NUM_EPISODES, Q)

if __name__=='__main__':
    main()       
    
