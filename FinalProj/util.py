import matplotlib.pyplot as plt
import numpy as np

def plot_ref_simu_traj(logs, save_path):
    """Plot the ref trajectory vs. controlled trajectory

    Args:
        logs: list of (desired trajectory, control trajectory)
        each trajectory contains a row matrix with each column a tuple of (time, x, y, theta)
        save_path: path to save the plot
    """
    labels = ['Desired trajectory','CL control trajectory']
    linestyles = ['-',':']
    colors = ['black','green']
    plt.figure()
    for i in range(len(logs)):
        log = logs[i]
        plt.plot(log[1,].T,log[2,].T,label = labels[i], linestyle = linestyles[i], color = colors[i])
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.legend(loc='best')
    plt.savefig(save_path)

def plot_cntl_traj(logs, save_path):
    # print(logs)
    plt.figure()
    plt.plot(logs[0],logs[1],linestyle = '-', color = 'black',label = 'Curvature control command')
    plt.plot(logs[0],logs[2],linestyle = ':', color = 'green',label = 'Curvature ego state')
    plt.xlabel('t[s]')
    plt.ylabel('curvature[1/m]')
    plt.legend(loc='best')
    plt.savefig(save_path)

def plot_actuator_modl_vs_gt(act_logs, save_path):
    plt.figure()
    labels = ['MPC model transition','MPC ground truth transition']
    linestyles = ['-',':']
    colors = ['red','green']
    plt.subplot(2,1,1)
    for i in range (2):
        plt.plot(act_logs[1][2],act_logs[1][i],label = labels[i], linestyle = linestyles[i], color = colors[i])
    plt.ylabel('curvature[1/m]')
    plt.legend(loc='best')
    plt.subplot(2,1,2)
    for i in range (2):
        plt.plot(act_logs[9][2],act_logs[9][i],label = labels[i], linestyle = linestyles[i], color = colors[i])
    plt.xlabel('t[s]')
    plt.ylabel('curvature[1/m]')
    plt.legend(loc='best')
    plt.savefig(save_path)

class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self,size_max):
        self.max = size_max
        self.data = np.zeros(self.max)
        self.cur = 0

    def append(self,x):

        """ Append an element overwriting the oldest one. """
        self.data[self.cur] = x

        self.cur = (self.cur+1) % self.max

    def get(self, idx):
        """ Return the idx oldest index """
        if abs(self.cur-1-idx) < len(self.data):
            value = self.data[self.cur-1-idx]
        else:
            raise Exception("idx large than max delay expected")

        return value


