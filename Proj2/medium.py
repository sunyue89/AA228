import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix

class Medium(object):
    def __init__(self,data,med_data,maxIter,learning_rate):
        self.States = np.arange(1,50001)
        N_States = np.size(self.States)
        self.Actions = np.unique(data[:,1])
        N_Action = np.size(self.Actions)
        self.NS = N_States
        self.NA = N_Action
        print(self.NS,self.NA)
        self.maxIter = maxIter
        self.U = np.zeros(N_States)
        self.policy = np.zeros(N_States)
        self.epsilon = 10
        self.Q = np.zeros((N_States,N_Action))
        self.alpha = learning_rate
        self.gamma = 1.0
        # uncomment to use value iteration and matrix inferral
        # self.R = self.make_reward_matrix()
        # self.data = med_data
        # self.TP = self.make_transition_matrix()
        # self.R = np.zeros((N_States,N_Action))

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

    def make_reward_matrix(self):
        R_1, R_2, R_3, R_4, R_5, R_6, R_7 = [np.zeros(self.NS) for _ in range(7)]
        R_1.fill(-225)
        R_2.fill(-100)
        R_3.fill(-25)
        R_4.fill(0)
        R_5.fill(-25)
        R_6.fill(-100)
        R_7.fill(-225)
        for arr in [R_1, R_2, R_3, R_4, R_5, R_6, R_7]:
            arr[[True if idx % 500 + 0.3 * idx // 500 > 475 else False for idx in range(1, 50001)]] = 100000
        return [R_1, R_2, R_3, R_4, R_5, R_6, R_7]

    def make_transition_matrix(self):
        transition_data = self.data.copy()
        transition_data = transition_data[transition_data.d_pos < 20] # re <move outliers
        transitions = []
        for a in range(1, 8):
            # print('generating T_{}...'.format(a))
            subset = transition_data[transition_data.a == a].copy()
            change_in_velocity = subset.groupby(subset.pos // 10).d_vel.apply(lambda x: x.value_counts() / len(x))
            change_in_position = subset.groupby(subset.vel // 10).d_pos.apply(lambda x: x.value_counts() / len(x))
            transition_matrix = dok_matrix((50000, 50000))
            for s in range(50000):

                pos = s % 500
                vel = s // 500

                # absorbing state
                if pos + 0.3 * vel > 475:
                    continue

                # hit wall
                if pos + 0.35 * vel <=16:
                    transition_matrix[s, 25000] += 1
                    continue

                pos_idx = pos // 10
                pos_idx = max(min(change_in_velocity.index)[0], pos_idx)
                pos_idx = min(max(change_in_velocity.index)[0], pos_idx)
                while pos_idx not in change_in_velocity:
                    pos_idx += 1
                dv_table = change_in_velocity[pos_idx]

                vel_idx = vel // 10
                vel_idx = max(min(change_in_position.index)[0], vel_idx)
                vel_idx = min(max(change_in_position.index)[0], vel_idx)
                while vel_idx not in change_in_position:
                    vel_idx += 1
                dp_table = change_in_position[vel_idx]

                for dp_pair in zip(dp_table.index, dp_table):
                    for dv_pair in zip(dv_table.index, dv_table):
                        dp, proba_dp = dp_pair
                        dv, proba_dv = dv_pair
                        sp = max(0, min((pos + dp) + (vel + dv) * 500, 49999))
                        transition_matrix[s, sp] += proba_dp * proba_dv

            transitions.append(transition_matrix.tocsr())
        return transitions

    def value_iteration(self):
        conv = False
        iter = 0
        print('value iteration starts...')
        while(conv == False and iter < self.maxIter):
            for a in range(self.NA):
                self.Q[:,a] = self.R[a] + self.TP[a].dot(self.U)
            self.U_last = np.copy(self.U)
            self.U = np.amax(self.Q,axis=1)
            # print(self.U)
            if max(self.U-self.U_last) < self.epsilon:
                conv = True
            iter += 1
            print(iter)
        self.policy = np.argmax(self.Q,axis=1)+1

    def output_policy(self):
        with open('medium.policy','w+') as f:
            f.writelines([str(x) + '\n' for x in self.policy])

    # def max_likelihood_est(self,data):
    #     #map data state and action to python friendly index
    #     # print(np.shape(data)[0])
    #     self.TP2D = np.zeros((self.N_States,self.N_States))
    #     self.TP = []
    #     for i in range(self.N_Action):
    #         self.TP.append(self.TP2D)
    #     # # print(self.TP)
    #     # # two dictionaries
    #     self.TPC = {}
    #     self.TC = {}
    #     for i in range (np.shape(data)[0]):
    #         s = data[i][0]-1
    #         a = data[i][1]-1
    #         sp = data[i][3]-1
    #         if self.TC.get((s,a)):
    #             self.TC[s,a] +=1
    #         else:
    #             self.TC[s,a] =1
    #         if self.TPC.get((s,sp,a)):
    #             self.TPC[s,sp,a] +=1
    #         else:
    #             self.TPC[s,sp,a] =1
    #         self.R[s,a] = data[i][2]
    #     for keys in self.TPC:
    #         self.TP[keys[2]][keys[0],keys[1]] = self.TPC[keys[0],keys[1],keys[2]]/self.TC[keys[0],keys[2]]
    #     print(self.TP)
    #     print('\n')