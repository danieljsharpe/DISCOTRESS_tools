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

    def __init__(self,k,stat,logvals):
        self.k = k
        self.stat = stat
        self.logvals = logvals
        self.action_vals = np.zeros(k,dtype=float)
        self.stat_vals = np.zeros(k,dtype=float)
        self.conv_vals = np.zeros(k,dtype=float)
        self.prob_sum = 0. # cumulative sum of first passage path probability

    def read_shortest_paths(self):
        with open("fpp_properties.dat") as fpp_f:
            for i, line in enumerate(fpp_f.readlines()):
                if i>=self.k: break # there are more entries in the output file than entries to be read in
                line = line.split()
                self.action_vals[i] = float(line[3])
                self.stat_vals[i] = float(line[self.stat])

    ''' use Kahan summation to obtain sum of probabilities for paths and convergence for weighted sum of path property,
        (hopefully) retaining numerical precision '''
    def calc_stat_convergence(self):
        self.prob_sum = 0.
        conv_sum = 0. # running total for weighted sum of path property
        c1 = 0. # running compensation for low-order floating points, for which accuracy is lost [path prob sum]
        c2 = 0. # another compensation value [path property weighted sum]
        for i in range(k):
            # sum of path probabilities
            e1 = np.exp(-1.*self.action_vals[i])
            y1 = e1-c1
            t1 = self.prob_sum+y1
            c1 = (t1-self.prob_sum)-y1
            self.prob_sum = t1
            # weighted sum of path properties
            if not self.logvals:
                e2 = np.exp(-1.*self.action_vals[i])*self.stat_vals[i]
            else:
                e2 = np.exp(-1.*self.action_vals[i])*(10.**self.stat_vals[i])
            y2 = e2-c2
            t2 = conv_sum+y2
            c2 = (t2-conv_sum)-y2
            conv_sum = t2
            self.conv_vals[i] = conv_sum
        if self.logvals: self.conv_vals = np.log10(self.conv_vals)

    def rea_plot(self,nxticks,nyticks1,ymin1,ymax1,nyticks2,ymin2,ymax2,mfppa,mfpt,ytick1_dp=2,ytick2_dp=0,figfmt="pdf"):
        if self.logvals: mfpt = np.log10(mfpt)
        xarr = np.arange(1,self.k+1,dtype=float)
        # first y axis (evolution of path action for successive shortest paths)
        fig, ax1 = plt.subplots(figsize=(10.,7.)) # fig size in inches
#        ax1.margins(0.5)
        line1=ax1.plot(xarr,self.action_vals,color="cornflowerblue",linewidth=6,label="$\mathrm{path\ action}$")
        ax1.set_xlabel("$\mathrm{Shortest\ path\ number}$",fontsize=42)
        ax1.set_ylabel("$-\ln{\mathcal{P}}\hspace{1}\mathrm{Path\ action}$",fontsize=42)
        ax1.hlines(mfppa,0,k,color="cornflowerblue",linewidth=6,linestyle="dashed")
        xtick_intvl=float(self.k)/float(nxticks)
        assert xtick_intvl.is_integer()
        ax1.set_ylim([ymin1,ymax1])
        xtick_vals = [0+(i*int(xtick_intvl)) for i in range(nxticks+1)]
        ytick1_intvl = (ymax1-ymin1)/float(nyticks1)
        ytick1_vals = [ymin1+(float(i)*ytick1_intvl) for i in range(nyticks1+1)]
        ax1.set_xticks(xtick_vals)
        ax1.set_yticks(ytick1_vals)
        xtick_labels=["$"+str(xtick_val)+"$" for xtick_val in xtick_vals]
        ytick1_labels=["$"+(("%."+str(ytick1_dp)+"f") % ytick_val)+"$" for ytick_val in ytick1_vals]
        ax1.set_xticklabels(xtick_labels)
        ax1.set_yticklabels(ytick1_labels)
        # second y axis (convergence of pathway sum for MFPT)
        ax2 = ax1.twinx() # second y axis shares same x axis as first y axis
        ax1.tick_params(which="both",direction="out",labelsize=24)
        line2=ax2.plot(xarr,self.conv_vals,color="deeppink",linewidth=6,label="$\mathrm{cumulative\ sum\ for\ MFPT}$")
        ax1.set_xlim(-float(xtick_intvl)/20.,self.k)
        if not self.logvals: MFPT_label = "MFPT"
        else: MFPT_label = "\log_{10}\mathcal{T}_{\mathcal{A}\mathcal{B}}"
        ax2.set_ylabel("$\mathrm{Path\ sum\ for\ "+MFPT_label+"}$",fontsize=42)
        plt.hlines(mfpt,0,k,color="deeppink",linewidth=6,linestyle="dashed")
        ax2.set_ylim([ymin2,ymax2])
        ax2.tick_params(labelsize=24)
        ytick2_intvl = (ymax2-ymin2)/float(nyticks2)
        ytick2_vals = [ymin2+(float(i)*ytick2_intvl) for i in range(nyticks2+1)]
        ax2.set_yticks(ytick2_vals)
        ytick2_labels=["$"+(("%."+str(ytick2_dp)+"f") % ytick_val)+"$" for ytick_val in ytick2_vals]
        ax2.set_yticklabels(ytick2_labels)
        # legend
        lines = line1+line2
        line_labels = [line.get_label() for line in lines]
        ax1.legend(lines,line_labels,loc="lower right",fontsize=20)
        # plot figure and save
        fig.tight_layout()
        plt.savefig("rea_plot."+figfmt,format=figfmt,bbox_inches="tight")
        plt.show()

if __name__=="__main__":

    ### SET PARAMS ###
    stat = 1 # path property to plot convergence of (1=time,2=length,3=action,4=entropy)
    k = 20000 # no. of shortest paths
    mfppa = 10. # mean first passage path action
    mfpt = 1.350008151E+22 # MFPT
    logvals = False # convergence of the path property specified by "stat" is plotted as log_10 values
    # plot parameters
    nxticks = 5
    nyticks1 = 3 # no. of ticks on first y-axis
    nyticks2 = 4 # no. of ticks on second y-axis
    ymin1 = 26.01   # min. on first y-axis (path action values for successive shortest paths)
    ymax1 = 26.04  # max. on first y-axis
    ymin2 = 5.   # min. on second y-axis (convergence of path property)
    ymax2 = 25. # max. on second y-axis

    assert(stat>0 and stat<5)
    ### RUN ###
    analyse_rea_obj = Analyse_REA(k,stat,logvals)
    analyse_rea_obj.read_shortest_paths()
    analyse_rea_obj.calc_stat_convergence()
    print("\n")
    print("total first passage path probability accounted for in %i highest-probability paths: %s" % (k,"{:.6e}".format(analyse_rea_obj.prob_sum)))
    print("pathway sum for MFPT from set of highest-probability paths: %s" % "{:.6e}".format(analyse_rea_obj.conv_vals[-1]))
#    analyse_rea_obj.rea_plot(nxticks,nyticks1,ymin1,ymax1,nyticks2,ymin2,ymax2,mfppa,mfpt)
