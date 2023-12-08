import numpy as np
import matplotlib.pyplot as plt

class Small(object):
    def __init__(self,data,discount,maxIter):
        N_States = np.size(np.unique(data[:,0]))
        N_Action = np.size(np.unique(data[:,1]))
        print(N_States,N_Action)
        self.NS = N_States
        self.NA = N_Action
        self.discount = discount
        self.U = np.zeros(N_States)
        self.policy = np.zeros(N_States)
        self.R = np.zeros((N_States,N_Action))
        self.TP = np.zeros((N_States,N_States,N_Action))
        self.epsilon = 0.01
        self.Q = np.zeros((N_States,N_Action))
        self.maxIter = maxIter

    def max_likelihood_est(self,data):
        for s1 in range (self.NS):
            idx1 = np.where(data[:,0]==s1+1)
            # print(idx1)
            ts = data[idx1,3]
            states = np.unique(ts)
            actions = data[idx1,1]
            for a in range (self.NA):
                idxAction = np.where(actions==a+1)
                ctAction = np.size(idxAction)
                #revisit
                i = np.intersect1d(np.where(data[:,0]==s1+1),np.where(data[:,1]==a+1))
                self.R[s1,a] = data[i[0],2]
                for s2 in states:
                    idx2 = np.where(ts[idxAction] == s2)
                    ctState = np.size(idx2)
                    self.TP[s1,s2-1,a] = ctState/ctAction

    def value_iteration(self):
        conv = False
        iter = 0
        while(conv == False and iter < self.maxIter):
            for a in range(self.NA):
                self.Q[:,a] = self.R[:,a] + self.discount*np.dot(self.TP[:,:,a],self.U)
            self.U_last = np.copy(self.U)
            self.U = np.amax(self.Q,axis=1)
            # print(self.U)
            self.policy = np.argmax(self.Q,axis=1)+1
            if max(self.U-self.U_last) < self.epsilon:
                conv = True
            iter += 1
        print(iter)


    def output_policy(self):
        with open('small.policy','w+') as f:
            f.writelines([str(x) + '\n' for x in self.policy])