# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:47:11 2020

@author: Aatish Rana
"""
import numpy as np 
from ann_visualizer.visualize import ann_viz

def sigmoid(x):
    return x > 0


class Perceptron:
    
    def __init__(self,n,m):
        '''Initialisation of the perceptron with sizes.'''
        
        self.input = np.ones(n+1)
        self.output = np.ones(m)
        self.weights = np.zeros((m,n+1))
        self.reset()
        
    def reset(self):
        '''Reset Weights'''
        
        Z = np.random.random(self.weights.shape)
        self.weights[...] = (2*Z-1)*0.25
        
    def propagate_forward(self, data):
        '''Propagate data from input layer to output layer. '''
        
        self.input[1:] = data
        self.output[...] = sigmoid(np.dot(self.weights, self.input))
        
        return self.output
    
    def propagate_backward(self ,target, lrate= 0.1):
        '''Back propagate error related to target using lrate'''
        
        error = np.atleast_2d(target-self.output)
        input = np.atleast_2d(self.input)
        self.weights+= lrate*np.dot(error.T, input)
        
        return (error**2).sum
    
if __name__ == '__main__':
    
    def learn(network, samples, epochs =250, lrate=.1, momentum=0.1):
        # Train
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward( samples['input'][n] )
            network.propagate_backward( samples['output'][n], lrate )
            
        # Test
        for i in range(samples.size):
            o = network.propagate_forward( samples['input'][i] )
            print(i, samples['input'][i], '%0.2f' %o[0])
            print('(expected %0.2f)' %samples['output'][i])
        print()
        
    network = Perceptron(2,1)
    samples = np.zeros(4, dtype=[('input',  float, 2), ('output', float, 1)])
    
    # Example 1 : OR logical function
    #-------------------------------------------------------------
    
    print("Learning the OR logical function")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (0,1), 1
    samples[2] = (1,0), 1
    samples[3] = (1,1), 1
    learn(network,samples)
    ann_viz(network)
    
    # Example 2 : AND logical function
    #-------------------------------------------------------------
    
    print("Learning the AND logical function")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (0,1), 0
    samples[2] = (1,0), 0
    samples[3] = (1,1), 1
    learn(network,samples)
    ann_viz(network)
    
    # Example 3 : XOR logical function
    #-------------------------------------------------------------
    
    print("Fails at learning the XOR logical function")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (0,1), 1
    samples[2] = (1,0), 1
    samples[3] = (1,1), 0
    learn(network,samples)
    ann_viz(network)

