#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Nov 11 16:06:53 2021

@author: pupulla

Issue

論文の方にはUCBの最初の全部の腕の探索がなかったので勝手に入れちゃった
regretの計算がおかしい説->t-1にして解決
Experimentが複数の場合 -> 解決
experimentを多くすると上界が小さくなる問題=>解決
複数のpolicyをplotするためには？->解決
Egreedyの変数Cはそもそもアルゴリズムはどうやって知るのか、元々知っていたら本末転倒では？ -> とりあえずなぜか知ってる設定


論文のと上界の数値が違いすぎ問題
時刻１のアームをラんダムに -> read me


"""
#%%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arm import Arms
from policy import (Policy, UCBPolicy, EgreedyPolicy, ThompsonPolicy)
import time


class Bandit(object):
    def __init__(self,ars:Arms, polic:Policy, l:float, label):
        self.arms = ars
        self.policy = polic
        self.l = l
        self.label = label
        
    def reset(self):
        self.arms.reset_arms()
        
        #cost, regret
    def drift(self,x):
        b = self.l*x
        return b
    
    def RelativeError(self):
        return abs((self.arms.mu_bars[0] - self.arms.mu_real[0]) / self.arms.mu_real[0] )
        
    def run(self, T:int, experiment):
        totalCost  = np.zeros((T,experiment))
        totalRegret= np.zeros((T,experiment))
        rError = np.zeros(experiment)
        countc = np.zeros(experiment)
        start_time = time.perf_counter()
        
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
                #print(It,Gt)
                #print(self.arms.mu_bars)
                #pulling It
                if It == Gt:
                    #without incentive
                    r = np.random.normal() + self.arms.mu_real[It]
                else:
                    #with incentive
                    countc[e] += 1
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
                    totalCost[t, e] += c
                    totalRegret[t,e] = self.arms.mu_real[0] - self.arms.mu_real[It]
                    
                else:
                    totalCost[t, e] = totalCost[t-1, e] + c
                    totalRegret[t,e] = totalRegret[t-1, e] +  self.arms.mu_real[0] - self.arms.mu_real[It]
                    
                t += 1
            rError[e] = self.RelativeError()
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time)/experiment
        print(self.label,"processing time :",elapsed_time)
        return np.mean(totalCost, axis=1), np.mean(totalRegret, axis=1), np.mean(countc), np.mean(rError)

    def single_plot(self, totalCost, totalRegret):
        sns.set_style('white')
        sns.set_context('talk')
        sns.set_palette('gray')
        plt.figure(figsize=(10, 10), dpi=50)
        plt.subplot(2, 1, 1)
        
        plt.title(self.label)
        plt.plot(totalCost)
        plt.ylim(0, 5000)
        plt.ylabel(' Cost')
        plt.xlabel('Time Step')
        
        plt.subplot(2, 1, 2)
        plt.plot(totalRegret)
        plt.ylim(0, 5000)
        plt.ylabel('Regret')
        plt.xlabel('Time Step')
        sns.despine()
        plt.savefig(self.label )
        plt.show()
        
    
    
    #%%
    
        
    
        
        