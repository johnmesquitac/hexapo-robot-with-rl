from q_learning_env import enviroment
from evaluate_training import evaluate_training
import matplotlib
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt 
import seaborn as sns
import random
import time
import math
import pandas as pd 

alpha = 0.99 # learning rate, learn more quickly if alpha is closer to 1
gamma = 0.95  # use a higher gamma for bigger spaces because we value later rewards rather than former rewards
epsilon_decay = 0.005
min_eps = 0.0
NUM_EPISODES = 300

env = enviroment()
#Q = np.zeros((env.states_size,env.actions_size)).tolist()
'''steps = []
rewards = []
penalties = []
states = []'''

def calculate_dynamic_reward(xold, yold, xnext, ynext, xend, yend, obstacle, begin):
    if begin == True:
        lambda_val = 100
    elif obstacle ==  True:
        lambda_val = 500
    else:
        lambda_val = 1

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

def select_optimal_action(state, Q):
    best = Q[state].index(max(Q[state]))
    return best

def take_next_step(env, done, state, epsilon,Q):
    old_x, old_y = env.get_state_index()
    env.insert_old_state_index(old_x, old_y)

    if np.random.random() < epsilon:
        action = env.select_random_action()
    else:
        action = select_optimal_action(state,Q)

    next_state,reward, done, obstacle, begin = env.next_step(action, state)

    next_x, next_y = env.get_state_index()
    goal_x, goal_y = env.get_goal_index()

    if done == False:
        reward_dynamic = calculate_dynamic_reward(old_x, old_y, next_x, next_y, goal_x, goal_y, obstacle, begin)
        reward += reward_dynamic
        
    next_best_action = np.argmax(Q[next_state])
    next_max_state_q_value = Q[next_state][next_best_action]
    Q[state][action] += alpha * (reward + gamma*next_max_state_q_value - Q[state][action])

    if obstacle==True:
        next_state = state

    return done, next_state, reward

def main():
    all_rewards = []
    all_time = []
    all_steps = []
    for i in range(10):
        steps = []
        rewards = []
        penalties = []
        states = []
        epsilon = 0.9 # agent can do 100% exploitation (0) or 100% exploration (1)
        steps, rewards, states = [], [], []
        Q = np.zeros((env.states_size,env.actions_size)).tolist()
        training_start = time.time()
        for i in range(NUM_EPISODES):
            print('episode', i)
            state, reward, done = env.reset_enviroment()
            step, total_reward = 0, 0
            while not done:
                done, state, reward = take_next_step(env, done, state, epsilon, Q)
                step+=1
                total_reward+=reward
                states.append(state)
            epsilon -= (epsilon_decay*epsilon) if epsilon>min_eps else 0
            steps.append(step)
            rewards.append(total_reward)
        training_end = time.time()
        #df = pd.DataFrame({'Passos':steps, 'Recompensas': rewards})
        #df.to_excel('recompensas_passos.xlsx')
        all_rewards.append(rewards)
        all_time.append(training_end-training_start)
        all_steps.append(steps)

    fig = plt.figure(figsize=(12,8))
    # Create the boxplot
    plt.boxplot(all_rewards, showfliers=False)
    ymin, ymax = plt.ylim()
    plt.yticks(np.arange(ymin,ymax, step=10))
    # Save the figure
    plt.xlabel('Estados')
    plt.ylabel('Recompensas')
    plt.title('Recompensas no Treinamento')
    plt.savefig("imgs/boxplot/boxplot_reward"+str((env.x*env.y)-1)+".png")

    fig = plt.figure(figsize=(12,8))
    # Create the boxplot
    plt.boxplot(all_steps, showfliers=False)
    ymin, ymax = plt.ylim()
    plt.yticks(np.arange(ymin,ymax, step=10))
    # Save the figure
    plt.xlabel('Estados')
    plt.ylabel('Passos')
    plt.title('Passos no Treinamento')
    plt.savefig("imgs/boxplot/boxplot_steps"+str((env.x*env.y)-1)+".png")

    print(all_time)


if __name__=='__main__':
    main()       
    
#[28.038007974624634, 30.193653345108032, 31.03703808784485, 29.769658088684082, 30.553129196166992, 30.923856019973755, 26.485406398773193, 27.8530855178833, 26.751949548721313, 28.024938583374023]