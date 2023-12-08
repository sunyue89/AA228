import sys
import logging
import pandas as pd
import time
import numpy as np
from small import Small
from medium import Medium
from large import Large


SMALL = './data/small.csv'
MEDIUM = './data/medium.csv'
LARGE = './data/large.csv'

def small(data,maxIter,discount):
    g = Small(data,discount,maxIter)
    start = time.time()
    g.max_likelihood_est(data)
    g.value_iteration()
    g.output_policy()
    end = time.time()
    print(end-start)

def medium(data,maxIter,learning_rate,Sarsa):
    med_data = pd.read_csv(MEDIUM)
    med_data['vel'] = med_data.s // 500
    med_data['pos'] = med_data.s % 500 - 1
    med_data['vel_p'] = med_data.sp // 500
    med_data['pos_p'] = med_data.sp % 500 - 1
    med_data['d_pos'] = med_data.pos - med_data.pos_p
    med_data['d_vel'] = med_data.vel - med_data.vel_p
    c = Medium(data,med_data,maxIter,learning_rate)
    start = time.time()
    # c.value_iteration()
    c.Q_learning(data,Sarsa)
    c.output_policy()
    end = time.time()
    print(end-start)

def large(data,discount,learning_rate,Sarsa):
    s = Large(data,discount,learning_rate)
    start = time.time()
    # True is Sarsa, False is Q Learning
    s.Q_learning(data,Sarsa)
    s.output_policy()
    end = time.time()
    print(end-start)

def load_data(file):
    data = pd.read_csv(file)
    data = data.to_numpy()
    # print(data)
    return data

def main():

    # data = load_data(SMALL)
    # small(data,10000,0.95)

    # data = load_data(MEDIUM)
    # medium(data,10000,0.9,False)

    data = load_data(LARGE)
    large(data,0.95,0.9,True)


if __name__ == '__main__':
    main()
