# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:02:38 2020

@author: Aatish Rana
"""

import numpy as np
from ann_visualizer.visualize import ann_viz
import matplotlib 
import matplotlib.pyplot as plt

def sigmoid(x):
    return np.tanh(x)

def diff_sigmoid(x):
    return 1.0 -x**2

'''Multiple layer perceptron'''
class MultipleLayerPerceptron:
    
    '''Initialization of the perceptron with given sizes'''
    
    def __init__(self, *args):
        
        self.shape = args
        n = len(args)

        #Build layers
        self.layers = []
        # Input layerv (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0]+1))
        #Hidden layers(s) +output layer
        
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))
            
        # Build weights matrix (randomly between -0.25 and +0.25)
        
        self.weights = []
        
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size, self.layers[i+1].size)))
            
        # dw will hold last change in weights (for momentum)
        
        self.dw = [0, ]*len(self.weights)
        
        # Reset Weights
        
        self.reset()
        
    def reset(self):
        '''Reset Weights'''
        
        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size, self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25
    
    def propagate_forward(self, data):
        '''Propagate data from input layer to output layer. '''
        
        # set the input layer 
        self.layers[0][0:-1] = data
        
        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        
        for i in range(1, len(self.shape)):
            # Propagate activity 
            
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1], self.weights[i-1]))
            
            
        # Return output
        
        return self.layers[-1]
    
    
    def propagate_backward(self, target, lrate=0.1,momentum=0.1):
        '''Back propagation error related to target using lrate'''
        
        deltas  = []
        
        # compute error on output layer
        error = target -self.layers[-1]
        delta = error*diff_sigmoid(self.layers[-1])
        deltas.append(delta)
        
        # Compute error on hidden layers
        for i in range(len(self.shape)-2, 0 , -1):
            delta = np.dot(deltas[0], self.weights[i].T)*diff_sigmoid(self.layers[i])
            deltas.insert(0,delta)
            
        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw
            
        # Return error 
        return (error**2).sum()
    
    
#----------------Main -----------------------------------------#
if __name__ == '__main__':
    
    def learn(network, samples, epochs = 2500, lrate = .1, momentum = 0.1):
        # Train
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward( samples['input'][n] )
            network.propagate_backward( samples['output'][n] , lrate, momentum)
        
        # Test
        for i in range(samples.size):
            o = network.propagate_forward( samples['input'][i])
            print (i, samples['input'][i], '%0.2f' % o[0])
            print('(expected %0.2f)' % samples['output'][i])
        print()
        
    
    
    network = MultipleLayerPerceptron(2,2,1)
    samples = np.zeros(4, dtype=[('input', float, 2), ('output', float, 1)])
    
    # Example 1 : OR logical function
    # -------------------------------------------------------------------------
    print("Learning the OR logical function")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 1
    samples[2] = (0,1), 1
    samples[3] = (1,1), 1
    learn(network, samples)

    # Example 2 : AND logical function
    # -------------------------------------------------------------------------
    print("Learning the AND logical function")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 0
    samples[2] = (0,1), 0
    samples[3] = (1,1), 1
    learn(network, samples)

    # Example 3 : XOR logical function
    # -------------------------------------------------------------------------
    print("Learning the XOR logical function")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 1
    samples[2] = (0,1), 1
    samples[3] = (1,1), 0
    learn(network, samples)

    # Example 4 : Learning sin(x)
    # -------------------------------------------------------------------------
    print("Learning the sin function")
    network = MultipleLayerPerceptron(1,5,1)
    samples = np.zeros(500, dtype=[('x',  float, 1), ('y', float, 1)])
    samples['x'] = np.linspace(0,1,500)
    samples['y'] = np.sin(samples['x']*np.pi)

    for i in range(10000):
        n = np.random.randint(samples.size)
        network.propagate_forward(samples['x'][n])
        network.propagate_backward(samples['y'][n])

    plt.figure(figsize=(10,5))
    # Draw real function
    x,y = samples['x'],samples['y']
    plt.plot(x,y,color='b',lw=1)
    # Draw network approximated function
    for i in range(samples.shape[0]):
        y[i] = network.propagate_forward(x[i])
    plt.plot(x,y,color='r',lw=3)
    plt.axis([0,1,0,1])
    plt.show()
    ann_viz(network)
    
    # Example 5 : Learning cos(x)
    # -------------------------------------------------------------------------
    print("Learning the cos function")
    network = MultipleLayerPerceptron(1,5,1)
    samples = np.zeros(500, dtype=[('x',  float, 1), ('y', float, 1)])
    samples['x'] = np.linspace(0,1,500)
    samples['y'] = np.cos(samples['x']*np.pi)

    for i in range(10000):
        n = np.random.randint(samples.size)
        network.propagate_forward(samples['x'][n])
        network.propagate_backward(samples['y'][n])

    plt.figure(figsize=(10,5))
    # Draw real function
    x,y = samples['x'],samples['y']
    plt.plot(x,y,color='b',lw=1)
    # Draw network approximated function
    for i in range(samples.shape[0]):
        y[i] = network.propagate_forward(x[i])
    plt.plot(x,y,color='r',lw=3)
    plt.axis([0,1,0,1])
    plt.show()