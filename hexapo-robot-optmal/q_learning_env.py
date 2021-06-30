import math
import numpy as np
import random

class enviroment:
    def __init__(self):
        self.x = 100
        self.y = 100
        self.x_position = 0
        self.y_position = 0
        self.goalx = self.x-1
        self.goaly = self.y-1
        self.action_space = [0, 1, 2, 3]
        self.actions_size = len(self.action_space)
        self.states_size = self.x*self.y
        self.state_matrix = np.ascontiguousarray(np.arange(self.states_size).reshape(self.x, self.y), dtype=int)
        self.matrix_list = self.state_matrix.tolist()

    def reset_enviroment(self):
        self.x_position = 0
        self.y_position = 0
        self.state = 0
        self.reward = 0
        return 0, 0, False
    
    def next_step(self, action):
        if self.x_position==self.goalx and self.y_position==self.goaly:
            done=True
            reward = 10
            return 15, reward, done
        else:
            # up
            if action==0 and self.x_position>0:
                self.x_position -= 1
            # Move left
            elif action== 1 and self.y_position>0:
                self.y_position-= 1
            # Move down
            elif action==2 and self.x_position<self.x-1:
                self.x_position+=1
            # Move right
            elif action==3 and self.y_position<self.y-1:
                self.y_position+=1
            done=False
            reward=-1
            next_state = self.state_matrix[self.x_position][self.y_position]
            return next_state, reward, done

    def get_state_index(self):
        return self.x_position, self.y_position
    
    def get_goal_index(self):
        return self.goaly, self.goaly

    def select_random_action(self):
        return np.random.choice(self.action_space)
