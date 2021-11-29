#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:11:19 2021

@author: pupulla
"""
#%%
import numpy as np
        
#時刻tのアーム群
#%%
class Arms(object):
    def __init__(self, mu_real):
        #とりあえず全部配列
        self.k = len(mu_real)
        self.ns = np.zeros(self.k) #pulling number array
        self.largeBs = np.zeros(self.k) #time t madeno drift no goukei array
        self.mu_real = mu_real
        self.mu_hats = np.zeros(self.k) #expected reward array
        self.mu_bars = np.zeros(self.k) #補正Expected reward array
        
    def reset_arms(self):
        """
        Resets the agent's memory to an initial state.
        """
        self.ns[:] = 0
        self.largeBs[:] = 0
        self.mu_hats[:] = 0
        self.mu_bars[:] = 0
        
        
    def update_mubar(self, i, rt,bt):
        """
        アームiのmu_barを更新、nは更新済み、tは未更新
        """
        self.mu_hats[i] = (self.mu_hats[i] * (self.ns[i] -1) +rt )/ self.ns[i] # new mu_hat
        self.largeBs[i] += bt # new largeB
        self.mu_bars[i] = self.mu_hats[i] + self.largeBs[i] / self.ns[i]
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    