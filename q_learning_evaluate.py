import numpy as np 
import pickle
from qlearning_training import select_optimal_action, reset_enviroment, initialize_state_matrix, identifiesgoal_state, identifies_state, next_step

action_space = np.array([0,1,2,3])


def select_optimal_path(q_table,enviroment):
    enviroment[0][0] = 1
    i,j = identifies_state(enviroment, enviromentsize) #reset enviroment to learn a new goal
    k, l = identifiesgoal_state(enviroment, enviromentsize)
    state = int(state_matrix[i][j])
    goal_state = int(state_matrix[k][l])
    states = []
    states.append(state)
    print(state, goal_state,'\n')
    print('\n', q_table, '\n', enviroment)
    old_state, reward, next_state = 0, 0, 0
    while(int(reward)<20):
        action = select_optimal_action(q_table, state)
        reward, next_state = next_step(q_table, enviroment, action, state, old_state, goal_state)
        old_state = state
        print(old_state, next_state)
        state = next_state
        states.append(state)
        print(states)
        print(state, goal_state,'\n', enviroment)
    print(states,'\n',q_table)

def main():
    global state_matrix, enviromentsize
    enviromentsize = 4
    state_size = enviromentsize*enviromentsize #defines environment
    env = np.zeros((enviromentsize, enviromentsize))
    env = reset_enviroment(env, enviromentsize)
    actions_size = enviromentsize #lef, right, up and down
    state_matrix = initialize_state_matrix(np.zeros((enviromentsize, enviromentsize)), enviromentsize)
    
    with open('q_table.pickle', "rb") as read:
        Q = pickle.load(read)

    select_optimal_path(Q,env)



if __name__ == '__main__':
    main()



