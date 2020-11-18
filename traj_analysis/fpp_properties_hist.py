'''
Python script to analyse the probability distributions of dynamical quantities in the first passage path ensemble. These statistics are
printed to the "fpp_properties.dat" output file from DISCOTRESS. Namely: the first passage time (FPT) distribution, and the path length,
path action (i.e. negative of log path probability), and path entropy flow distributions.
The script plots a histogram of the probability distribution and calculates the mean and variance of the distribution (and the associated
standard errors) for the chosen first passage path property.

Daniel J. Sharpe
Jan 2020
'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from math import sqrt

class Analyse_fpp_properties(object):

    def __init__(self,stat,nbins,binw,bin_min,binall,logvals):
        if stat<1 or stat>4: raise RuntimeError
        self.stat=stat
        self.nbins=nbins
        self.binw=binw
        self.bin_min=bin_min
        self.bin_max=self.bin_min+(self.nbins*self.binw)
        self.binall=binall
        self.logvals=logvals
        self.ntpaths=0
        self.vals=None

    def get_hist_arr(self):
        hist_arr = np.zeros(self.nbins,dtype=int)
        vals=[]
        with open("fpp_properties.dat","r") as pathprops_f:
            for line in pathprops_f.readlines():
                val=float(line.split()[stat])
                if self.logvals: val=np.log10(val)
                vals.append(val)
                if not (val>=self.bin_max or val<self.bin_min):
                    hist_arr[int(floor((val-self.bin_min)/self.binw))] += 1
                elif self.binall:
                    print("found bad value: ",val,"for path: ",self.ntpaths+1)
                    raise RuntimeError
                self.ntpaths+=1
        self.vals=np.array(vals,dtype=float)
        return hist_arr

    def plot_hist(self,hist_arr,nxticks,nyticks,ymax,fpd_name,figfmt="pdf",color="cornflowerblue",xtick_dp=0,ytick_dp=2,
                  linevals=None,linecolor="deeppink"):
        hist_arr=hist_arr.astype(np.float64)*1./float(self.ntpaths) # normalise
        bins=[self.bin_min+(i*self.binw) for i in range(self.nbins)]
        plt.figure(figsize=(10.,7.)) # size in inches
        plt.bar(bins,hist_arr,self.binw,color=color,edgecolor=color)
        if self.logvals:
            plt.xlabel("$\log_{10}("+fpd_name+")$",fontsize=42)
            plt.ylabel("$p ( \log_{10} ("+fpd_name+")$",fontsize=42)
        else:
            plt.xlabel("$"+fpd_name+"$",fontsize=42)
            plt.ylabel("$p("+fpd_name+")$",fontsize=42)
        if linevals is not None:
            plt.vlines(linevals,0.,ymax,colors=linecolor,linewidths=6.,linestyles="dashed")
        plt.savefig("fp_distribn."+figfmt,format=figfmt,bbox_inches="tight")
        ax = plt.gca()
        ax.set_xlim([self.bin_min,self.bin_max])
        ax.set_ylim([0.,ymax])
        ax.tick_params(direction="out",labelsize=24)
        xtick_intvl=float(self.bin_max-self.bin_min)/float(nxticks)
        ytick_intvl=float(ymax)/float(nyticks)
        xtick_vals=[self.bin_min+(float(i)*xtick_intvl) for i in range(nxticks+1)]
        if xtick_intvl.is_integer(): xtick_vals = [int(xtick_val) for xtick_val in xtick_vals]
        ytick_vals=[0.+(float(i)*ytick_intvl) for i in range(nyticks+1)]
        ax.set_xticks(xtick_vals)
        ax.set_yticks(ytick_vals)
        xticklabels=["$"+(("%."+str(xtick_dp)+"f") % xtick_val)+"$" for xtick_val in xtick_vals]
        yticklabels=["$"+(("%."+str(ytick_dp)+"f") % ytick_val)+"$" for ytick_val in ytick_vals]
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        plt.tight_layout()
        plt.savefig("fp_distribn."+figfmt,format=figfmt,bbox_inches="tight")
        plt.show()

    ''' calculate mean of first passage time (FPT) distribution '''
    def calc_mfpt(self):
        if not self.logvals: return np.sum(self.vals)/float(self.ntpaths)
        else: return np.sum([10**val for val in self.vals])/float(self.ntpaths)

    ''' calculate variance of first passage time (FPT) distribution '''
    def calc_var_fptd(self):
        if not self.logvals:
            var = (np.sum(np.array([val**2 for val in self.vals]))/float(self.ntpaths))-((np.sum(self.vals)/float(self.ntpaths))**2)
        else:
            var = (np.sum(np.array([(10**val)**2 for val in self.vals]))/float(self.ntpaths))-\
                  ((np.sum(np.array([10**val for val in self.vals]))/float(self.ntpaths))**2)
        return var

    ''' calculate standard error associated with the MFPT. (Alternatively, =sqrt(var)/sqrt(n)) '''
    def calc_stderr_mfpt(self,mfpt):
        stderr=0.
        for val in self.vals:
            if not self.logvals: stderr+=(val-mfpt)**2
            else: stderr+=(10**val-mfpt)**2
        return sqrt((1./float(self.ntpaths-1))*stderr)/sqrt(float(self.ntpaths))

if __name__=="__main__":
    ### CHOOSE PARAMS ###

    # statistic to analyse
    # 1=time, 2=dynamical activity (path length), 3=-ln(path prob) [path action], 4=entropy flow
    stat=1

    # binning params

    nbins=50
    binw=0.1
    bin_min=0.  # 18.
    binall=False # enforce that all values must be encompassed in the bin range
    logvals=True # take log_10 of values
    # plot params
    nxticks=10 # no. of ticks on x axis
    nyticks=10 # no. of ticks on y axis
    ymax=0.1 # max value for y (prob) axis
    # can add one or more vertical lines to plot (e.g. to indicate mean value)
    linevals = np.array([3646.89402349])

    # run
    calc_hist_obj=Analyse_fpp_properties(stat,nbins,binw,bin_min,binall,logvals)
    hist_arr = calc_hist_obj.get_hist_arr()
    print("\nhistogram bin counts:\n",hist_arr)
    print("\ntotal number of observed A<-B transition paths:\t",calc_hist_obj.ntpaths)
    print("total number of binned A<-B transition paths:\t",np.sum(hist_arr))
    mfpt = calc_hist_obj.calc_mfpt()
    var_fptd = calc_hist_obj.calc_var_fptd()
    std_err = calc_hist_obj.calc_stderr_mfpt(mfpt)
    std_dev=sqrt(var_fptd)
    print("\nmean of FPT distribution (MFPT):\t",mfpt)
    print("variance of FPT distribution:\t\t",var_fptd)
    print("standard error in MFPT:\t\t\t",std_err)
    print("standard error in var:\t\t\t",var_fptd*sqrt(2./(calc_hist_obj.ntpaths-1.)))
    # plot
    if logvals: linevals = np.log10(linevals)
    fpd_name=None
    if stat==1: fpd_name = "t_\mathrm{FPT}"
    elif stat==2: fpd_name = "\mathcal{L}"
    elif stat==3: fpd_name = "- \ln \mathcal{P}"
    elif stat==4: fpd_name = "\mathcal{S} / k_\mathrm{B}"
    else: quit("error in choice of stat")
    calc_hist_obj.plot_hist(hist_arr,nxticks,nyticks,ymax,fpd_name,figfmt="svg",xtick_dp=1,ytick_dp=3,linevals=linevals)
