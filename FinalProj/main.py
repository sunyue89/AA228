import numpy as np
import math
import sys
import copy
import time
import ctrl
import plnt
import util
import learn

class Simu:
    def __init__(self, tau, Td) -> None:
        self.Tau =  tau
        self.Td  =  Td
        self.L   =  3
        self.V   =  5
        self.dT  =  0.01
        # Ref time
        self.TR = 16
        # Simu time
        self.T = 12
        # Control horizon
        self.Tc = 1
        self.N = 10
        self.x_mpc = np.zeros((3,self.N+1))
        self.x_gt = np.zeros((3,self.N+1))
        self.x_dt = np.zeros((1,self.N+1))
        # Initial state
        self.X_init = np.array([[0.0],[0.0],[0.0]])
        self.Qm = np.zeros((3,3))
        self.Qm[0,0] = 5.0
        self.Qm[1,1] = 0.2
        self.Qm[2,2] = 0
        self.Rm = np.eye(1)*0.1
        # Desired trajectory with a Radius of 10 [m] at speed V = 5[m/s]
        # Lateral acc = V^2/R = 2.5 [m/ss]
        self.R = 20
        self.ref_traj = np.r_[np.array([[0.0]]),self.X_init]
        X = np.array([[0.0],[0.0],[0.0]])
        # Iteration steps
        N = int(self.TR/self.dT)
        for i in range(0,N):
            t = np.array([[(i+1)*self.dT]])
            dOmega = self.V/self.R
            X[2] += dOmega*self.dT
            X[0] += math.cos(X[2])*self.V*self.dT
            X[1] += math.sin(X[2])*self.V*self.dT
            self.ref_traj = np.c_[self.ref_traj, np.r_[t,X]]
        self.state_array = np.array([[0.0],[0.0],[0.0],[0.0]])
        self.ref_array = np.array([[0.0],[0.0],[0.0],[0.0]])
        #tau, Td, L, V, dT, Q, R, T, N
        self.cntrl = ctrl.Cntrl(self.Tau, self.Td, self.L, self.V, self.dT, self.Qm, self.Rm, self.Tc, self.N)
        #tau, Td, V, dT, Xinit
        self.plant = plnt.Plant(self.Tau, self.Td, self.V, self.dT, self.X_init)
        self.logs = []
        self.logs.append(self.ref_traj)
        self.simu_log = np.r_[np.array([[0.0]]),self.X_init]
        self.u_log = np.r_[[0.0],[0.0],[0.0]]
        self.act_log = []

    def reset(self):
        self.state_array = np.array([[0.0],[0.0],[0.0],[0.0]])
        self.ref_array = np.array([[0.0],[0.0],[0.0],[0.0]])
        #print(self.X_init)
        self.plant.reset(self.X_init)
        #no need to reset control
        self.simu_log = np.r_[np.array([[0.0]]),self.X_init]
        self.logs = []
        self.logs.append(self.ref_traj)

    def simulate_one_step(self, tau, Td, tau_true, Td_true):
        self.ref_local = self.plant.coord_txform_global_local(self.ref_traj,self.state_array)
        # print('state',self.state_array)
        # print('\n')
        # print(self.ref_traj)
        # print('\n')
        # print(self.ref_local)
        # print('\n')
        # At this moment, assume the controller is aware of the ground-truth low-level system dynamics
        # To be replaced by the model based learning function
        # tau = copy.deepcopy(tau_true)
        # Td = copy.deepcopy(Td_true)
        U, self.x_mpc, self.x_gt, self.x_dt = self.cntrl.compute_cntrl_output(self.ref_local,self.plant.u,tau,Td,tau_true,Td_true)
        self.plant.euler_plant(tau_true, Td_true, U)
        self.state_array[0] += self.dT
        self.state_array[1:] = self.plant.X
        self.simu_log = np.c_[self.simu_log, self.state_array]
        # print(self.state_array[0],U,self.plant.u)
        self.u_log = np.c_[self.u_log,np.r_[self.state_array[0],U,self.plant.u]]

    def simulate(self,tau_true, Td_true, learn_flag):
        done = False
        t0 = -1
        T = 2
        while not done:
            start = time.time()
            self.simulate_one_step(self.Tau, self.Td, tau_true, Td_true)
            if (self.state_array[0,0] - t0) > self.Tc:
                t0 = self.state_array[0,0]
                log = [self.x_mpc[2,:],self.x_gt[2,:],self.x_dt[0,:]]
                self.act_log.append(log)
                # print(t0,log,self.act_log)
                if t0 <= self.Tc+0.1:
                    TypicalCycleTime = time.time() - start
                else:
                    TypicalCycleTime = 0.5 * TypicalCycleTime + 0.5*(time.time() - start)

            if learn_flag == True and self.state_array[0] > 1.5:
                print('Learning Cycle\n')
                learner = learn.Learner(self.u_log)
                x_init = [1.0, self.Tau, self.Td]
                x = learner.learn(x_init)
                self.Tau = x[1]
                self.Td = x[2]
                learn_flag = False
                LearnCycleTime = time.time() - start

            if self.state_array[0] > self.T:
                print(LearnCycleTime,TypicalCycleTime)
                done = True
        self.logs.append(self.simu_log)
        self.plot('Sim_ref_vs_cntrl','Sim_u_vs_t','MPC_model_vs_gt')

    def plot(self,path1,path2,path3):
        util.plot_ref_simu_traj(self.logs,path1)
        util.plot_cntl_traj(self.u_log,path2)
        util.plot_actuator_modl_vs_gt(self.act_log,path3)


if __name__ == '__main__':
    #200[ms] time constant
    tau = 0.2
    #200[ms] delay
    Td = 0.2
    sim = Simu(tau, Td)
    learnFlag = True
    sim.simulate(0.6, 0.3, learnFlag)