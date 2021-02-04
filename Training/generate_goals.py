import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import random
import pickle
from time import time
from matplotlib import pyplot as plt


def initialize_state_matrix():
    cont = 0
    for i in range(enviromentsize):
        for j in range(enviromentsize):
            state_matrix[i][j] = int(cont)
            if int(cont) in goals:
                state_matrix[i][j] = 20
            elif i == 0 and j == 0:
                state_matrix[i][j] = 1
            else:
                state_matrix[i][j] = 0
            cont += 1
    if int(decision) == 2:
        state_matrix[1][1] = -1
        state_matrix[1][2] = -1
        state_matrix[2][2] = -1


def plot_matrix():
    figure, ax = plt.subplots()
    im = ax.matshow(state_matrix)
    for i in range(enviromentsize):
        for j in range(enviromentsize):
            text = ax.text(
                j, i, state_matrix[i, j], ha="center", va="center", color="w")
    figure.tight_layout()
    plt.show()


def return_back(lista):
    for i in range(len(lista)):
        if lista[i] == 'R':
            lista[i] = 'L'
        elif lista[i] == 'L':
            lista[i] = 'R'
        elif lista[i] == 'U':
            lista[i] = 'D'
        elif lista[i] == 'D':
            lista[i] = 'U'
    return reversed(lista)


def defining_steps():
    steps = []
    steps_going_back = []
    for i in range(int(goal_steps)):
        if int(decision) == 2:
            if goals[i] < 5:
                steps.append(Q[goals[i]-1])
                steps.append(reversed(Qcopy[goals[i]-1]))
            elif goals[i] > 4 and goals[i] < 10:
                print(Q[goals[i]-3])
                steps.append(Q[goals[i]-3])
                steps.append(reversed(Qcopy[goals[i]-3]))
            elif goals[i] > 10 and goals[i] < 16:
                steps.append(Q[goals[i]-4])
                steps.append(reversed(Qcopy[goals[i]-4]))
        elif int(decision) == 1:
            steps.append(Q[goals[i]-1])
            steps.append(reversed(Qcopy[goals[i]-1]))
    for lista in steps:
        for element in lista:
            steps_going_back.append(element)
    print(steps_going_back)


def defining_return_steps():
    for element in Qcopy:
        element = return_back(element)


def main():
    global Q, goals, enviromentsize, state_matrix, goal_steps, Qcopy, decision
    enviromentsize = 4
    goals = []
    decision = input(
        "Please press 1 if you want to work with an enviroment without obstacles or press 2 to work in an enviroment with obstacles \n")
    if int(decision) == 1:
        with open('pickle/without_obstacles/steps_positions.pickle', "rb") as read:
            Q = pickle.load(read)
        with open('pickle/without_obstacles/steps_positions.pickle', "rb") as read:
            Qcopy = pickle.load(read)
    elif int(decision) == 2:
        with open('pickle/with_obstacles/steps_positions.pickle', "rb") as read:
            Q = pickle.load(read)
        with open('pickle/with_obstacles/steps_positions.pickle', "rb") as read:
            Qcopy = pickle.load(read)

    goal_steps = input("How many goal positions do you need?\n")
    print("Please enter with the goal positions\n")
    for _ in range(int(goal_steps)):
        x = input()
        goals.append(int(x))
    state_matrix = np.zeros((enviromentsize, enviromentsize))
    initialize_state_matrix()
    defining_return_steps()
    defining_steps()
    plot_matrix()


if __name__ == '__main__':
    main()
