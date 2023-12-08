import numpy as np
import matplotlib.pyplot as plt

class Large(object):
    def __init__(self,data,discount,learningRate):
        self.States = np.arange(1, 312021)
        N_States = np.size(self.States)
        self.Actions = np.arange(1, 10)
        N_Action = np.size(self.Actions)
        self.NS = N_States
        self.NA = N_Action
        print(self.NS,self.NA)
        self.gamma = discount
        self.policy = np.zeros(N_States)
        self.alpha = learningRate
        self.Q = np.zeros((N_States,N_Action))
        self.U = np.zeros(N_States)


    def Q_learning(self,data,Sarsa):
        for i in range (np.shape(data)[0]):
            if (i==np.shape(data)[0]-1 or data[i][3] != data[i+1][0]):
                continue
            s = data[i][0]-1
            a = data[i][1]-1
            r = data[i][2]
            sp = data[i][3]-1
            ap = data[i+1][1]-1
            self.Q_last = np.copy(self.Q)
            if Sarsa:
                self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[sp,ap] - self.Q[s, a])
            else:
                self.Q[s, a] += self.alpha * (r + self.gamma * max(self.Q[sp]) - self.Q[s, a])
        self.U = np.amax(self.Q,axis=1)
        # print([i for i in self.U if i!=0])
        for s in range(self.NS):
            if not np.any(self.Q[s]):
                self.policy[s] = np.random.randint(1,self.NA+1)
            else:
                self.policy[s] = np.argmax(self.Q[s]) + 1
        Convergence = np.linalg.norm(self.Q_last - self.Q)
        print(Convergence)


    def output_policy(self):
        with open('large.policy','w+') as f:
            f.writelines([str(x) + '\n' for x in self.policy])
