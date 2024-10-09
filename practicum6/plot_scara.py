# Import library
import math
import matplotlib.pyplot as plt
import numpy as np

class ScaraPlot(object):
    def __init__(self, model, L, corner):

        # Target coordinates of square
        n = 100 # number of steps
        x = np.array([ np.linspace(0,L, n),  np.linspace(L,L, n), np.linspace(L,0,n), np.linspace(0,0, n),]).reshape(4*n)
        x = x+corner[0]
        y = np.array([np.linspace(0, 0,n), np.linspace(0,L, n), np.linspace(L,L, n),  np.linspace(L, 0, n)]).reshape(4*n)
        y = y+corner[1]
        
        self.square_x = x
        self.square_y = y
        self.square_n = n
        
        # Get model predictions
        EE = np.array([self.square_x,self.square_y]).T
        q_pred = (model.predict(EE))

        # Input Arm length
        self.l1 = 0.5
        self.l2 = 0.5

        # Define Angle variable
        self.theta_1 = q_pred[:,0]
        self.theta_2 = q_pred[:,1]

        # Input Position of (x0,y0)
        self.x0 = 0
        self.y0 = 0

        # Compute all frames
        self.positions_x1 = []
        self.positions_y1 = []
        self.positions_x2 = []
        self.positions_y2 = []
        self.num_frames = self.theta_1.shape[0]

        for f in range(self.num_frames):
            # Calculate coordinates (x1, y1)
            x1 = self.l1 * math.cos(self.theta_1[f])
            y1 = self.l1 * math.sin(self.theta_1[f])
            self.positions_x1.append(x1)
            self.positions_y1.append(y1)

            # Calculate (x2,y2)
            x2 = x1 + self.l2*math.cos(self.theta_1[f] + self.theta_2[f])
            y2 = y1 + self.l2*math.sin(self.theta_1[f] + self.theta_2[f])
            self.positions_x2.append(x2)
            self.positions_y2.append(y2)

    def plot_frame(self, f):
        #plt.figure(1)
        #plt.clf()

        # Plot end effector path
        #plt.scatter(self.positions_x2[:f], self.positions_y2[:f], color='C1')
        plt.plot(self.positions_x2[:f], self.positions_y2[:f], color='C1')

        # Plot axis limit
        plt.axis('equal')
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])

        # Plot square
        x = self.square_x
        y = self.square_y
        n = self.square_n
        plt.plot([x[0], x[n-1], x[2*n-1], x[3*n-1], x[4*n-1]], [y[0], y[n-1], y[2*n-1], y[3*n-1], y[4*n-1]], '--')

        # Plot of robotics arm
        x0 = self.x0
        y0 = self.y0
        x1 = self.positions_x1[f]
        y1 = self.positions_y1[f]
        x2 = self.positions_x2[f]
        y2 = self.positions_y2[f]
        
        plt.plot([x0,x1], [y0,y1],'r', linewidth=10)
        plt.plot([x1,x2], [y1,y2],'b', linewidth=10)
        #plt.show()
