'''
Python script to plot the results of the WRAPPER REA method in DISCOTRESS to compute the k highest probability paths using the recursive enumeration algorithm
The script reads the file "fpp_properties.dat", plots path action as a function of k, and plots convergence of MFPT with k

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
                self.action_vals[i] = float(line[2])
                self.stat_vals[i] = float(line[self.stat])

    def calc_stat_convergence(self):
        self.conv_vals[0] = np.exp(-1.*self.action_vals[0])*self.stat_vals[0]
        for i in range(1,k):
            self.conv_vals[i] = self.conv_vals[i-1]+(np.exp(-1.*self.action_vals[0])*self.stat_vals[0])

    def rea_plot(self,nxticks,nyticks,ymin,ymax,mfpt,mfppa,ytick_dp=1,figfmt="pdf"):
        xarr = np.arange(1,self.k+1,dtype=float)
        plt.figure(figsize=(10.,7.)) # fig size in inches
#        plt.plot(xarr,self.action_vals,color="blue")
        plt.plot(xarr,self.conv_vals,color="deeppink",linewidth=12)
        plt.xlabel("$\mathrm{Shortest\ path\ number}$",fontsize=42)
        plt.ylabel("$\mathrm{Path\ action}\ -\ln{\mathcal{P}}$",fontsize=42)
        plt.hlines(mfpt,0,k,color="deeppink",linewidth=6,linestyle="dashed")
        ax = plt.gca()
        ax.set_xlim([0,self.k])
        ax.set_ylim([ymin,ymax])
        ax.tick_params(direction="out",labelsize=24)
        xtick_intvl=float(self.k)/float(nxticks)
        assert xtick_intvl.is_integer()
        xtick_vals = [0+(i*int(xtick_intvl)) for i in range(nxticks+1)]
        ytick_intvl = (ymax-ymin)/float(nyticks)
        ytick_vals = [ymin+(float(i)*ytick_intvl) for i in range(nyticks+1)]
        ax.set_xticks(xtick_vals)
        ax.set_yticks(ytick_vals)
        xtick_labels=["$"+str(xtick_val)+"$" for xtick_val in xtick_vals]
        ytick_labels=["$"+(("%."+str(ytick_dp)+"f") % ytick_val)+"$" for ytick_val in ytick_vals]
        ax.set_xticklabels(xtick_labels)
        ax.set_yticklabels(ytick_labels)
        plt.tight_layout()
        plt.savefig("rea_plot."+figfmt,format=figfmt,bbox_inches="tight")
        plt.show()

if __name__=="__main__":

    ### SET PARAMS ###
    stat = 0 # path property to plot convergence of (0=time,1=length,2=action,3=entropy)
    k = 1000 # no. of shortest paths
    mfpt = 1.499218 # MFPT
    mfppa = 1.5 # mean first passage path action
    # plot parameters
    nxticks = 10
    nyticks = 10
    ymin = 0.
    ymax = 2.

    ### RUN ###
    analyse_rea_obj = Analyse_REA(k,stat)
    analyse_rea_obj.read_shortest_paths()
    analyse_rea_obj.calc_stat_convergence()
    print(analyse_rea_obj.conv_vals[900:1000])
    analyse_rea_obj.rea_plot(nxticks,nyticks,ymin,ymax,mfpt,mfppa)
