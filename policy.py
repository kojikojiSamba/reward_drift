#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:06:53 2021

@author: pupulla

armの情報
self.k = len(mu_real) #number of arms
self.n = np.zeros(self.k) #pulling number array
self.largeB = np.zeros(self.k) #time t madeno drift no goukei array
self.mu_hat = np.zeros(self.k) #expected reward array
self.mu_bar = np.zeros(self.k) #補正Expected reward array
"""
#%%
import numpy as np
from arm import Arms
#%%
class Policy(object):
    """
    A policy prescribes an action to be taken based on the memory of an agent.
    """
    def __str__(self):
        return 'generic policy'

    def choose(self, arm, t:int):
        return 0
#%%
class UCBPolicy(Policy):
    """
    The Upper Confidence Bound algorithm (UCB1). It applies an exploration
    factor to the expected value of each arm which can influence a greedy
    selection strategy to more intelligently explore less confident options.
    """
    def __init__(self):
        pass
    
    def choose(self, arms:Arms, t:int):
        
        if t < arms.k:
            return t
        else:
            ucbScore = [arms.mu_bars[i] + np.sqrt(2*np.log(t+1) / arms.ns[i]) for i in range(arms.k)]
            It = np.argmax(ucbScore)
            return It
        """
        ucbScore = [arms.mu_bars[i] + np.sqrt(2*np.log(t+1) / arms.ns[i]) for i in range(arms.k)]
        It = np.argmax(ucbScore)
        return It
        """
    
#%%
class EgreedyPolicy(Policy):
    """
    The Epsilon-Greedy policy will choose a random action with probability
    epsilon and take the best apparent approach with probability 1-epsilon. If
    multiple actions are tied for best choice, then a random action from that
    subset is selected.
    """
    def __init__(self, arms:Arms):
        #self.c = 36 / np.min([arms.mu_real[0] - arms.mu_real[i] for i in range(1, arms.k)])
        self.c =100
    def choose(self, arms:Arms, t:int):
        #update c -> no need update every time
        epsilon = min(1, (self.c*arms.k)/(t+1))
        if np.random.random() < epsilon:
            return np.random.randint(arms.k)
        else:
            return np.argmax(arms.mu_bars)
#%%
class ThompsonPolicy(Policy):  
    def __init__(self):
        pass
    
    def choose(self, arms:Arms,t:int):
        thmary = [np.random.normal(loc=arms.mu_bars[i], scale= 1/(arms.ns[i]+1), size=None) for i in range(arms.k)]
        It = np.argmax(thmary)
        return It
#%%

    