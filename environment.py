#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Nov 11 16:06:53 2021

@author: pupulla

Issue

論文の方にはUCBの最初の全部の腕の探索がなかったので勝手に入れちゃった
regretの計算がおかしい説->t-1にして解決
Experimentが複数の場合 -> 解決

Egreedyの変数Cはそもそもアルゴリズムはどうやって知るのか、元々知っていたら本末転倒では？
複数のpolicyをplotするためには？
時刻１のアームをラんダムに
ucb全探索なしをやる
egreedy cのlower boundを時刻毎に更新

"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from arm import Arms
from policy import (Policy, UCBPolicy, EgreedyPolicy, ThompsonPolicy)







class Bandit(object):
    def __init__(self,ars:Arms, polic:Policy, label):
        self.arms = ars
        self.policy = polic
        self.label = label
        
    def reset(self):
        self.arms.reset_arms()
        
        #cost, regret
    def drift(self,x):
        l = 1
        b = l*x
        return b
        
    def run(self, T:int, experiment):
        totalCost  = np.zeros((T,experiment))
        totalRegret= np.zeros((T,experiment))
        for e in range(experiment):
            self.reset()
            t=0
            while(t<T):
                # 実際の時刻はt+1
                It = self.policy.choose(self.arms, t) #principal choose arm
                Gt = np.argmax(self.arms.mu_bars) # player choose arm
                
                r = 0
                b = 0
                c=0
                print(It,Gt)
                #pulling It
                if It == Gt:
                    #without incentive
                    r = np.random.normal() + self.arms.mu_real[It]
                else:
                    #with incentive
                    c = self.arms.mu_bars[Gt] - self.arms.mu_bars[It]
                    b = self.drift(c)
                    r = np.random.normal() + self.arms.mu_real[It] + b
                    
                """
                update ni, mu_bar/hat(i), t
                calcurate total regret, total cost
                """
                self.arms.ns[It] += 1
                self.arms.update_mubar(It, r, b)

                if t ==0:
                    totalCost[t, experiment-1] += c
                    totalRegret[t,experiment-1] = self.arms.mu_real[0] - self.arms.mu_real[It]
                    
                else:
                    totalCost[t, experiment-1] = totalCost[t-1, experiment-1] + c
                    totalRegret[t,experiment-1] = totalRegret[t-1, experiment-1] +  self.arms.mu_real[0] - self.arms.mu_real[It]
                    
                t += 1
            print(e)
        
        return np.mean(totalCost, axis=1), np.mean(totalRegret, axis=1)
    
    def plot(self, totalCost, totalRegret):
        sns.set_style('white')
        sns.set_context('talk')
        sns.set_palette('gray')
        
        plt.subplot(2, 1, 1)
        plt.title(self.label)
        plt.plot(totalCost)
        plt.ylim(0, 30)
        plt.ylabel(' Cost')
        plt.xlabel('Time Step')
        
        plt.subplot(2, 1, 2)
        plt.plot(totalRegret)
        plt.ylim(0, 200)
        plt.ylabel('Regret')
        plt.xlabel('Time Step')
        sns.despine()
        plt.show()
        #plt.savefig(self.label + 'png')
    #%%
    
        
    
        
        