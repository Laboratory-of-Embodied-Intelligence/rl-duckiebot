import torch 
import os
import numpy as numpy

class OUNoise(object):
    
    def __init__(self, mu, sigma = 0.15, theta = 0.2, dt = 1e-2, x0=None):
        """ Initialize functor with noise
        
        Arguments:
            mu {float} -- mu param
        
        Keyword Arguments:
            sigma {float} -- sigma param (default: {0.15})
            theta {float} -- theta param (default: {0.2})
            dt {[type]} -- dt param (default: {1e-2})
            x0 {[type]} -- starting x0 (default: {None})
        """
        self.t = theta
        self.m = mu
        self.s = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
        
    def __call__(self):
        x = self.x_prev + self.t * (self.m - self.x_prev) * self.dt + self.s * np.sqrt(self.dt) * np.random.normal(size=self.m.shape)
        
        self.x_prev = x
        return x
   
    def reset(self):
        self.x_prev = self.x0 if self.x0 else np.zeros_like(self.m)
    
    def __repr__(self):
        return (f"Ornstein Uhlenbeck Noise with mu={self.m} sigma{self.s}")


