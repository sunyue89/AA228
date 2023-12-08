import numpy as np
import math
import util

class Plant:
    """Plant model deploys a kinematic veh model + 1st steering actuator model
        System input as curvature command and output as the three veh states
        System dynamics are based on global coord system (X,Y,Theta) thus no need of coord transformation
        States integration are based on forward Euler
        The initial pose is assumed to start at (0,0,0) w/o explicit pass in of Xinit
    """
    def __init__(self, tau, Td, V, dT, Xinit) -> None:
        self.Tau = tau
        self.Td = Td
        self.u = 0
        self.V = V
        self.dT = dT
        self.X = Xinit.copy()
        self.dX = np.array([[0.0],[0.0],[0.0]])
        self.Ubuf = util.RingBuffer(int(0.5/self.dT))
        self.lat_acc = 0
        self.lat_acc_last = 0
        self.lat_jerk = 0

    def reset(self, Xinit):
        self.X = Xinit.copy()
        self.dX = np.array([[0.0],[0.0],[0.0]])

    def euler_plant(self, tau, Td, U):
        """The state space model integrator that deploys forward euler
        """
        if tau > 0 and self.Tau != tau:
            self.Tau = tau
        if Td > 0 and self.Td != Td:
            self.Td = Td
        self.Ubuf.append(U)
        u_delay = self.Ubuf.get(int(self.Td/self.dT))
        self.u = self.u*(1-self.dT/self.Tau) + u_delay*self.dT/self.Tau
        # print(u_delay,self.u)
        # print('\n')
        # self.u = self.u*(1-self.dT/self.Tau) + U*self.dT/self.Tau
        self.dX[2] = self.u*self.V*self.dT
        self.dX[1] = math.sin(self.X[2])*self.V*self.dT
        self.dX[0] = math.cos(self.X[2])*self.V*self.dT
        self.X[2] += self.dX[2]
        self.X[1] += self.dX[1]
        self.X[0] += self.dX[0]
        self.lat_acc = 0.99*self.V*self.dX[2] + 0.01*self.lat_acc_last
        self.lat_jerk = (self.lat_acc - self.lat_acc_last)/self.dT
        self.lat_acc_last = self.lat_acc

    def coord_txform_global_local(self,ref,X):
        """Take the reference and delta ego pose (both global), transform and return reference based on ego body frame
            in addition, the function will also return heading and lateral error at the current ego pose
            ref will be provided as a row matrix of tuples, with each column tuple represented by (t,X,Y,Theta)
            dX will be provided as a tuple represented by (dX, dY, dTheta)
            dT will be provided as a scalar representing the time difference (minus) between two adjacent time steps
        """
        d_theta = -X[3]
        RtMatrix = np.array(
            [[math.cos(d_theta), -math.sin(d_theta),0],
                [math.sin(d_theta), math.cos(d_theta),0],
                [0,                 0,                1]])
        local = np.empty(ref.shape)
        for i in range(ref.shape[1]):
            d_x = ref[1,i] - X[1]
            d_y = ref[2,i] - X[2]
            # TxMatrix = np.array(
            # [[1, 0, d_x],
            #  [0, 1, d_y],
            #  [0, 0, 1]])
            local[0,i] = ref[0,i] - X[0]
            local[1:,i] = np.dot(RtMatrix,np.array([d_x,d_y,1.0]))
            local[3,i] = ref[3,i] - X[3]
        # print('ac',local[0:,-1])
        # print('bc',ref[0:,-1])
        # print('X',X)
        return local