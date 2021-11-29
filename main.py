#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:52:21 2021

@author: pupulla
"""
import numpy as np
from arm import Arms
from policy import (Policy, UCBPolicy, EgreedyPolicy, ThompsonPolicy)
from environment import Bandit as bd


if __name__ == '__main__':
    experiment = 10
    trials = 150
    
    mu_real = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    arms = Arms(mu_real)
    policy = UCBPolicy()
    label = 'UCB Policy'

    env = bd(arms, policy, label)
    cost, regret = env.run(trials, experiment)
    #%%
    env.plot(cost, regret)