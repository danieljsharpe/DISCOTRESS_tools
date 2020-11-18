'''
Python script to plot the results of the WRAPPER REA method in DISCOTRESS to compute the k highest-probability paths using
the recursive enumeration algorithm.
The script reads the file "fpp_properties.dat", plots path action values as a function of k, and plots the convergence of
the MFPT with k.

Daniel J. Sharpe
Nov 2020
'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


class Analyse_REA(object):

    def __init__(self,k,stat):
        self.k = k
        self.stat = stat
        self.action_vals = np.zeros(k,dtype=float)
        self.stat_vals = np.zeros(k,dtype=float)
        self.conv_vals = np.zeros(k,dtype=float)

    def read_shortest_paths(self):
        with open("fpp_properties.dat") as fpp_f:
            for i, line in enumerate(fpp_f.readlines()):
                line = line.split()
                self.action_vals[i] = float(line[3])
                self.stat_vals[i] = float(line[self.stat])

    def calc_stat_convergence(self):
        self.conv_vals[0] = np.exp(-1.*self.action_vals[0])*self.stat_vals[0]
        for i in range(1,k):
            self.conv_vals[i] = self.conv_vals[i-1]+(np.exp(-1.*self.action_vals[0])*self.stat_vals[0])

    def rea_plot(self,nxticks,nyticks1,ymin1,ymax1,nyticks2,ymin2,ymax2,mfppa,mfpt,ytick_dp=1,figfmt="pdf"):
        xarr = np.arange(1,self.k+1,dtype=float)
        # first y axis (evolution of path action for successive shortest paths)
        fig, ax1 = plt.subplots(figsize=(10.,7.)) # fig size in inches
        ax1.plot(xarr,self.action_vals,color="cornflowerblue",linewidth=6)
        ax1.set_xlabel("$\mathrm{Shortest\ path\ number}$",fontsize=42)
        ax1.set_ylabel("$-\ln{\mathcal{P}}\hspace{1}\mathrm{Path\ action}$",fontsize=42)
        ax1.hlines(mfppa,0,k,color="cornflowerblue",linewidth=6,linestyle="dashed")
        ax1.set_xlim([0,self.k])
        ax1.set_ylim([ymin1,ymax1])
        ax1.tick_params(direction="out",labelsize=24)
        xtick_intvl=float(self.k)/float(nxticks)
        assert xtick_intvl.is_integer()
        xtick_vals = [0+(i*int(xtick_intvl)) for i in range(nxticks+1)]
        ytick1_intvl = (ymax1-ymin1)/float(nyticks1)
        ytick1_vals = [ymin1+(float(i)*ytick1_intvl) for i in range(nyticks1+1)]
        ax1.set_xticks(xtick_vals)
        ax1.set_yticks(ytick1_vals)
        xtick_labels=["$"+str(xtick_val)+"$" for xtick_val in xtick_vals]
        ytick1_labels=["$"+(("%."+str(ytick_dp)+"f") % ytick_val)+"$" for ytick_val in ytick1_vals]
        print(ytick1_labels)
        ax1.set_xticklabels(xtick_labels)
        ax1.set_yticklabels(ytick1_labels)
        # second y axis (convergence of pathway sum for MFPT)
        ax2 = ax1.twinx() # second y axis shares same x axis as first y axis
        ax2.plot(xarr,self.conv_vals,color="deeppink",linewidth=6)
        ax2.set_ylabel("$\mathrm{Path\ sum\ for\ MFPT}$",fontsize=42)
        plt.hlines(mfpt,0,k,color="deeppink",linewidth=6,linestyle="dashed")
        ax2.set_ylim([ymin2,ymax2])
        ax2.tick_params(direction="out",labelsize=24)
        ytick2_intvl = (ymax2-ymin2)/float(nyticks2)
        ytick2_vals = [ymin2+(float(i)*ytick2_intvl) for i in range(nyticks2+1)]
        ax2.set_yticks(ytick2_vals)
        ytick2_labels=["$"+(("%."+str(ytick_dp)+"f") % ytick_val)+"$" for ytick_val in ytick2_vals]
        ax2.set_yticklabels(ytick2_labels)
        # plot figure and save
        fig.tight_layout()
        plt.savefig("rea_plot."+figfmt,format=figfmt,bbox_inches="tight")
        plt.show()

if __name__=="__main__":

    ### SET PARAMS ###
    stat = 1 # path property to plot convergence of (1=time,2=length,3=action,4=entropy)
    k = 25000 # no. of shortest paths
    mfppa = 10. # mean first passage path action
    mfpt = 7089.44 # MFPT
    # plot parameters
    nxticks = 5
    nyticks1 = 10 # no. of ticks on first y-axis
    nyticks2 = 10 # no. of ticks on second y-axis
    ymin1 = 5.   # min. on first y-axis (path action values for successive shortest paths)
    ymax1 = 15.  # max. on first y-axis
    ymin2 = 0.   # min. on second y-axis (convergence of path property)
    ymax2 = 10000. # max. on second y-axis

    assert(stat>0 and stat<5)
    ### RUN ###
    analyse_rea_obj = Analyse_REA(k,stat)
    analyse_rea_obj.read_shortest_paths()
    analyse_rea_obj.calc_stat_convergence()
    print(analyse_rea_obj.conv_vals[:50])
    analyse_rea_obj.rea_plot(nxticks,nyticks1,ymin1,ymax1,nyticks2,ymin2,ymax2,mfppa,mfpt)
