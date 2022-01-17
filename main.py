#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:52:21 2021

@author: pupulla
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
from arm import Arms
from policy import (Policy, UCBPolicy, EgreedyPolicy, ThompsonPolicy)
from environment import Bandit as bd

def logApp(x,a,b):
    y = a + b * np.log(x)
    return y

def deg1App(x,a,b):
    y = a + b * x
    return y

def deg2App(x,a,b,c):
    y = a + b * x + (c * x * x)
    return y

def app(x_ary, y_ary,labbel,typ):
    if(typ == 0):
        a_pred,b_pred = curve_fit(logApp, x_ary, y_ary)[0]
        print(labbel,"est:",a_pred,"/",b_pred)
    elif(typ == 1):
        a_pred,b_pred = curve_fit(deg1App, x_ary, y_ary)[0]
        print(labbel,"est:",a_pred,"/",b_pred)
    else:
        a_pred,b_pred,c_pred = curve_fit(deg2App, x_ary, y_ary)[0]
        print(labbel,"est:",a_pred,"/",b_pred,"/", c_pred)
    

if __name__ == '__main__':
    experiment = 100
    trials = 20000
    
    mu_real = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    arms = Arms(mu_real)
    l = 1.1
    ls = [0, 0.05, 0.1, 0.4, 0.7, 1.0, 1.1 ]
    #%%
    
    env2 = bd(arms, ThompsonPolicy(), l,'Thompson Policy_0.png' )
    cost2, regret2 = env2.run(trials, experiment)
    env2.single_plot(cost2, regret2)
    
    #%%
    
    """
    comparison between different coefficients
    """
    
    lll = len(ls)
    """
    Rt=np.zeros(lll)
    Ct=np.zeros(lll)regret EP : [1493.818 1486.054 1531.91  1512.858 1590.495 4474.94  5841.775]
    cmp of ep: [ 1587.36628826  1433.3338701   1307.95426698   885.96136598
       827.72076884  5046.49280745 11044.10357105]
    regret Thompson : [861.124 491.361 596.629 409.362 363.51  412.497 303.226]
    cmp of thompson: [23.89948498 23.77693874 24.23259608 25.3554942  26.27705788 25.59195686
     25.43603959]
    """
    rt_ucb=np.zeros(lll)
    ct_ucb=np.zeros(lll)
    rt_ep=np.zeros(lll)
    ct_ep=np.zeros(lll)
    rt_th=np.zeros(lll)
    ct_th=np.zeros(lll)
    i = 0
    while(i != lll):
        
        env1 = bd(arms, UCBPolicy(), ls[i],'UCB Policy' )
        ctu, rtu,_,_, = env1.run(trials, experiment)
        rt_ucb[i] = rtu[trials-1]
        ct_ucb[i] = ctu[trials-1]
        
        env2 = bd(arms, EgreedyPolicy(arms), ls[i],'egreedy Policy' )
        cte, rte,_,_, = env2.run(trials, experiment)
        rt_ep[i] = rte[trials-1]
        ct_ep[i] = cte[trials-1]
        
        env3 = bd(arms, ThompsonPolicy(), ls[i], 'thompson Policy' )
        ctt, rtt,_,_, = env3.run(trials, experiment)
        rt_th[i] = rtt[trials-1]
        ct_th[i] = ctt[trials-1]
        
        i +=1
        
    print("regret of UCB :", rt_ucb)
    app(ls, rt_ucb, "UCB reg l", 2)
    print("cmp of tucb:", ct_ucb)
    app(ls, ct_ucb, "UCB cmp l", 1)
    """
    print("regret EP :", rt_ep)
    print("cmp of ep:", ct_ep)
    print("regret Thompson :", rt_th)
    print("cmp of thompson:", ct_th)
    """
    #%%
    #複数比較する用
    x = np.arange(1,trials+1)
    env1 = bd(arms, UCBPolicy(), l,'UCB Policy' )
    cost1, regret1, _,_ = env1.run(trials, experiment)
    app(x, cost1, "UCB compensation")
    app(x, regret1, "UCB regret")
    
    env2 = bd(arms, EgreedyPolicy(arms), l,'egreedy Policy' )
    cost2, regret2,_,_= env2.run(trials, experiment)
    app(x, cost2, "Epsilon compensation")
    app(x, regret2, "Epsilon regret")
    
    env3 = bd(arms, ThompsonPolicy(), l, 'thompson Policy' )
    cost3, regret3, _,_ = env3.run(trials, experiment)
    app(x, cost3, "TS compensation")
    app(x, regret3, "TS regret")
    
    sns.set_style('white')
    sns.set_context('talk')
    sns.set_palette('gray')
    
    
    plt.figure(figsize=(10, 10), dpi=50)
    plt.subplot(2, 1, 1)
    plt.plot(cost1, "g", label="UCB")
    plt.plot(cost2, "b", label="epsilon greedy")
    plt.plot(cost3,"r", label="Thompson Sampling")
    plt.legend(bbox_to_anchor=(1,1), loc="upper right")
    plt.ylim(0, 15000)
    plt.ylabel(' Compensation C(T)')
    plt.xlabel('Time Step T')
    
    plt.subplot(2, 1, 2)
    plt.plot(regret1, "g", label="UCB1")
    plt.plot(regret2, "b", label="epsilon greedy")
    plt.plot(regret3,"r", label="Thompson")
    plt.legend(bbox_to_anchor=(1,1), loc="upper right")
    plt.ylim(0, 15000)
    plt.ylabel('Regret R(T)')
    plt.xlabel('Time step T')
    sns.despine()
    plt.savefig("l11_e100_20k_5.png")
    plt.show()

    