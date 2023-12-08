import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import util

class Learner:
    def __init__(self, data) -> None:
        """
        data is a 2D numpy array with the following row entries
        time(t), input(u), output(yp)
        """
        self.t = data[0,:]-data[0,0]
        self.u = data[1,:]
        self.y = data[2,:]
        self.u0 = self.u[0]
        self.y0 = self.y[0]
        self.ns = len(self.t)
        self.delta_t = self.t[1]-self.t[0]
        self.uf = interp1d(self.t,self.u)

    def update(self, data) -> None:
        """
        """
        self.t = data[0,:]-data[0,0]
        self.u = data[1,:]
        self.y = data[2,:]
        self.u0 = self.u[0]
        self.y0 = self.y[0]
        self.ns = len(self.t)
        self.delta_t = self.t[1]-self.t[0]

    def fopdt(self,y,t,uf,Km,taum,Tm):
        # y    = output
        # t    = time
        # uf   = input linear function
        # Km   = model gain
        # taum = model time constant
        # Tm   = time delay
        try:
            # print(t,Tm)
            if (t-Tm) <= 0:
                um = uf(0.0)
            else:
                um = uf(t-Tm)
        except:
            um = self.u0
        # calculte derivative
        dydt = (-(y-self.y0) + Km*(um-self.u0))/taum
        return dydt

    def sim_model(self,x):
        # input parameters
        Km = x[0]
        taum = x[1]
        Tm = x[2]
        # storage for model values
        ym = np.zeros(self.ns)
        # initial condition
        ym[0] = self.y0
        # loop through time steps
        for i in range(0,self.ns-1):
            ts = [self.t[i],self.t[i+1]]
            # print(ts)
            y1 = odeint(self.fopdt,ym[i],ts,args=(self.uf,Km,taum,Tm))
            ym[i+1] = y1[-1]
        return ym

    def objective(self,x):
        ym = self.sim_model(x)
        obj = 0.0
        for i in range(len(ym)):
            obj += (ym[i]-self.y[i])**2
        return obj

    def learn(self,xinit):
        #another way to solve - with bounds on variables
        bnds = ((0.95, 1.05), (0.1, 1.0), (0.0, 0.5))
        solution = minimize(self.objective, xinit, bounds = bnds, method = 'SLSQP')
        # print(solution)
        self.debug(xinit,solution.x)
        self.plot(xinit,solution.x,'Learner_Model_Learned')
        return solution.x

    def debug(self,xinit,x):
        # show final objective
        print('First SSE Objective: ' + str(self.objective(xinit)))
        print('Final SSE Objective: ' + str(self.objective(x)))

        print('Kp: ' + str(x[0]))
        print('taup: ' + str(x[1]))
        print('thetap: ' + str(x[2]))

    def plot(self,xinit,x,save_path):
        # calculate model with updated parameters
        ym1 = self.sim_model(xinit)
        ym2 = self.sim_model(x)
        # plot results
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.t,self.y,'kx-',linewidth=2,label='Process Data')
        plt.plot(self.t,ym1,'b-',linewidth=2,label='Initial Guess')
        plt.plot(self.t,ym2,'r--',linewidth=3,label='Optimized FOPDT')
        plt.ylabel('Output')
        plt.legend(loc='best')
        plt.subplot(2,1,2)
        plt.plot(self.t,self.u,'bx-',linewidth=2)
        # plt.plot(self.t,uf(t),'r--',linewidth=3)
        # plt.legend(['Measured','Interpolated'],loc='best')
        plt.ylabel('Input Data')
        plt.xlabel('Input Time [s]')
        plt.savefig(save_path)
