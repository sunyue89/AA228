import numpy as np
import math
import copy
import sys
import cvxpy as cp
import util

class Cntrl:
    """
    """
    def __init__(self, tau, Td, L, V, dT, Q, R, T, N) -> None:
        self.Tau = tau
        self.Td = Td
        self.Tau_true = tau
        self.Td_true = Td
        self.Q = Q
        self.R = R
        self.V = V
        self.n = 3
        self.m = 1
        self.dT = dT
        self.A = np.array([[0, V, 0],
                           [0, 0, V],
                           [0, 0, -1/tau]])
        self.B = np.array([[0],[0],[1/tau]])
        self.Ad = self.A*self.dT + np.eye(3)
        self.Bd = self.B*self.dT
        self.A_true = np.array([[0, V, 0],
                           [0, 0, V],
                           [0, 0, -1/tau]])
        self.B_true = np.array([[0],[0],[1/tau]])
        self.Ad_true = self.A_true*self.dT + np.eye(3)
        self.Bd_true = self.B_true*self.dT
        self.Umax = 30*math.pi/180/L
        self.Udotmax = 0.20
        #max delay is 500[ms]
        self.Ubuf = util.RingBuffer(int(0.5/self.dT))
        self.N = N
        self.T = T
        self.MPC_dT = self.T/self.N

    def cntrl_para_update(self,tau,Td,tau_true,Td_true):
        if tau > 0 and self.Tau != tau:
            self.Tau = tau
        if Td > 0 and self.Td != Td:
            self.Td = Td
        if tau_true > 0 and self.Tau_true != tau_true:
            self.Tau_true = tau_true
        if Td_true > 0 and self.Td_true != Td_true:
            self.Td_true = Td_true
        self.A[2,2] = -1/self.Tau
        self.B[2] = 1/self.Tau
        self.Ad = self.A*self.dT + np.eye(3)
        self.Bd = self.B*self.dT
        self.A_true[2,2] = -1/self.Tau_true
        self.B_true[2] = 1/self.Tau_true
        self.Ad_true = self.A_true*self.dT + np.eye(3)
        self.Bd_true = self.B_true*self.dT

    def MPC_solve(self,x_init,ref):
        x = cp.Variable((self.n,self.N+1))
        u = cp.Variable((self.m,self.N))
        constr = [x[:,0] == x_init]
        # constr = [cp.abs(u[:,0] - self.Ubuf.get(int(self.MPC_dT/self.dT))) <= self.Udotmax*self.MPC_dT]
        idx_shift = int(self.Td/self.MPC_dT)
        cost = 0
        for i in range(self.N):
            if i >= idx_shift:
                constr += [x[:,i+1] == self.Ad@x[:,i] + self.Bd@u[:,i-idx_shift]]
                constr += [u[:,i-idx_shift] <= self.Umax]
                constr += [cp.abs(u[:,i-idx_shift+1] - u[:,i-idx_shift]) <= self.Udotmax*self.MPC_dT]
                cost_one_term = cp.quad_form(x[:,i],self.Q) - 2*ref[:,i].T@self.Q@x[:,i] + cp.quad_form(u[:,i-idx_shift],self.R)
                cost += cost_one_term
                # print(x[:,i],u[:,i-idx_shift])
            else:
                u_delay = cp.Constant(self.Ubuf.get(int((self.Td-self.MPC_dT*i)/self.dT))*np.ones((self.m)))
                constr += [x[:,i+1] == self.Ad@x[:,i] + self.Bd@u_delay]
                # print(x[:,i],u_delay)


        cp.Problem(cp.Minimize(cost),constr).solve()
        self.Ubuf.append(u.value[:,0])
        # print('x_init',x_init)
        # print('\n')
        # print('ref',ref)
        # print('\n')
        # print('control',u.value[:,0])
        # print('\n')
        # print('MPC_traj',x.value[:,1:])
        # print('\n')
        return u.value, x.value

    def MPC_ground_truth(self,x_init,MPC_u):
        GT_x = np.zeros((self.n,self.N+1))
        x_dT = np.zeros((1,self.N+1))
        GT_x[:,0] = copy.deepcopy(x_init)
        idx_shift = int(self.Td_true/self.MPC_dT)
        for i in range(self.N):
            x_dT[0,i+1] = self.MPC_dT*(i+1)
            if i >= idx_shift:
                GT_x[:,i+1] = self.Ad_true@GT_x[:,i] + self.Bd_true@MPC_u[:,i-idx_shift]
            else:
                u_delay = self.Ubuf.get(int((self.Td_true-self.MPC_dT*i)/self.dT))*np.ones((self.m))
                GT_x[:,i+1] = self.Ad_true@GT_x[:,i] + self.Bd_true@u_delay
        return GT_x, x_dT

    def MPC_init_and_ref_lookup(self,ref_local,curv):
        x_init = np.zeros(3)
        MPC_ref = np.zeros((3,self.N))
        init_set = False
        j = 0
        t0 = 100000000
        # time, dx, dy, d_theta
        # print('ref_local',ref_local)
        for i in range(ref_local.shape[1]):
            if ref_local[1,i] >= 0 and init_set == False:
                x_init[0] = ref_local[2,i] #dy
                x_init[1] = ref_local[3,i] #dtheta
                x_init[2] = curv
                t0 = ref_local[0,i]
                init_set = True
            if (ref_local[0,i] - t0) >= self.MPC_dT*(j+1) and j<self.N:
                # print(j)
                MPC_ref[0,j] = ref_local[2,i] #dy
                MPC_ref[1,j] = ref_local[3,i] #dtheta
                # MPC_ref[2,j] = np.arctan2(ref_local[2,i],ref_local[1,i]) #trivial
                j += 1
        return x_init, MPC_ref

    def compute_cntrl_output(self,ref_local,curv,tau,Td,tau_true,Td_true):
        self.cntrl_para_update(tau,Td,tau_true,Td_true)
        x_init, MPC_ref = self.MPC_init_and_ref_lookup(ref_local,curv)
        MPC_u, MPC_x = self.MPC_solve(x_init, MPC_ref)
        GT_x, dT_x = self.MPC_ground_truth(x_init,MPC_u)
        return MPC_u[:,0], MPC_x, GT_x, dT_x