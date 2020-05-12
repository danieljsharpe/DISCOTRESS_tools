'''
Python code to read the output of a DISCOTRESS "WRAPPER DIMREDN" simulation (many short trajectories of fixed time, initialised from each community
in turn), and estimate and validate a coarse-grained Markov chain. Functionality:
  -estimate a reversible coarse-grained discrete-time Markov chain with fixed stationary distribution by maximum-likelihood or Gibbs sampling methods (requires PyEMMA)
  -estimate a reversible coarse-grained continuous-time Markov chain with fixed stationary distribution by maximum-likelihood method (uses a modified version of some MSMBuilder scripts, no requirements)
  -perform implied timescale, Chapman-Kolmogorov and correlation function tests

Reads from files:
 walker.x.y.dat, communities.dat, stat_prob.dat, ntrajs.dat (refer to DISCOTRESS documentation for more detail)

References:
[1] D. J. Sharpe and D. J. Wales, Dimensionality reduction of Markovian networks using efficient dynamical simulations (in preparation)
[2] B. Trendelkamp-Schroer, H. Wu, F. Paul, and F. Noe, J. Chem. Phys. 143, 174101 (2015).
[3] B. Trendelkamp-Schroer and F. Noe, J. Chem. Phys. 138, 164113 (2013).
[4] R. T. McGibbon and V. S. Pande, J. Chem. Phys. 143, 034109 (2015).
[5] M. K. Schrerer et al., J. Chem. Theory Comput. 11, 5525-5542 (2015).
[6] M. P. Harrigan et al., Biophys. J. 112, 10-15 (2017).
[7] J.-H. Prinz et al., J. Chem. Phys. 134, 174105 (2011).
[8] N.-V. Buchete and G. Hummer,  J. Phys. Chem. B 112, 6057-6069 (2008).
    
If you use these methods, please cite the relevant publications:
-dimensionality reduction of Markov chains with DISCOTRESS: [1]
-maximum-likelihood estimation of reversible DTMCs with fixed stationary distribution: [2], [5]
-Gibbs sampling of reversible DTMCs with fixed stationary distribution: [2], [3], [5]
-maximum-likelihood estimations of reversible CTMCs: [4], [6]
-Chapman-Kolmogorov and implied timescale tests: [7]
-correlation function test: [8]


Daniel J. Sharpe
May 2020
'''  

import numpy as np
import matplotlib.pyplot as plt
import pyemma.msm as msm
from msmbuilder.msm import ContinuousTimeMSM
from ctmc_mle_obj import CTMC_MLE_Obj
from scipy.linalg import expm
from math import floor
from math import ceil
import subprocess

''' class to read trajectory data output from an accelerated kMC simulation using DISCOTRESS,
    and perform estimation and validatation of a coarse-grained Markov chain '''
class Discotress_coarsegrainer(object):

    def __init__(self,n_macrostates,tau):
        self.n_macrostates = n_macrostates
        self.tau = tau
        self.pi = Discotress_coarsegrainer.read_single_col("stat_prob.dat",float)
        self.comms = Discotress_coarsegrainer.read_single_col("communities.dat",int)
        self.calc_pi_coarse()
        self.dtrajs = None

    ''' calculate the implied timescales for a series of MLE transition probability matrices estimated at integer multiples of tau
        this information can be used to find the optimum lag time at which to estimate the transition matrices '''
    def calc_implied_timescales(self,dlag_min,dlag_max,dlag_intvl=1,n_timescales=None):
        if n_timescales is None: n_timescales=self.n_macrostates-1 # compute implied timescales for all states (note stationary implied timescale is ignored)
        else: assert n_timescales<self.n_macrostates # compute specified number of dominant implied timescales (ignoring stationary implied timescale)
        print "calculating %i dominant implied timescales with lag times in range: (%f,%f) at intervals: %f" % \
              (n_timescales,float(dlag_min)*self.tau,float(dlag_max)*self.tau,float(dlag_intvl)*self.tau)
        implt_arr = np.zeros((n_timescales,(dlag_max-dlag_min+1)/dlag_intvl),dtype=float)
        dlag_vec = np.array([dlag for dlag in range(dlag_min,dlag_max+1,dlag_intvl)])
        for i, dlag in enumerate(range(dlag_min,dlag_max+1,dlag_intvl)):
            T, pi = self.estimate_dtmc_mle(dlag)
            T_eigs, T_evecs = np.linalg.eig(T)
            T_eigs = np.array(sorted(list(T_eigs),reverse=True),dtype=float)
            for j in range(n_timescales):
                implt_arr[j,i] = (-1.*float(dlag)*tau)/np.log(T_eigs[j+1])
        return implt_arr, dlag_vec

    ''' plot the implied timescales to check for convergence '''
    def implied_timescales_plot(self,implt_arr,dlag_vec,figfmt="pdf"):
        print "plotting implied timescales"
        implt_arr=np.log10(implt_arr)
        plt.figure(figsize=(10.,7.)) # size in inches
        for i in range(np.shape(implt_arr)[0]):
            plt.plot(dlag_vec,implt_arr[i,:],linewidth=5)
        plt.xlabel("Lag time $\\tau\ /\ t_\mathrm{intvl}$",fontsize=42)
        plt.ylabel("Implied timescale $\log_{10} t_k$",fontsize=42)
        ax = plt.gca()
        ax.set_xlim([dlag_vec[0],dlag_vec[-1]])
        miny, maxy = floor(np.amin(implt_arr)), ceil(np.amax(implt_arr))
        nyticks=11
        ax.set_ylim([miny,maxy])
        ytick_intvl=(maxy-miny)/float(nyticks-1)
        ytick_vals=[miny+(float(i)*ytick_intvl) for i in range(nyticks)]
        ax.tick_params(direction="out",labelsize=24)
        ax.set_xticks(dlag_vec)
        ax.set_yticks(ytick_vals)
        xticklabels=["$"+str(dlag)+"$" for dlag in dlag_vec]
        yticklabels=["$"+str(ytick_val)+"$" for ytick_val in ytick_vals]
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        plt.tight_layout()
        plt.savefig("tk_test."+figfmt,format=figfmt,bbox_inches="tight")
        plt.show()

    ''' propagate an initial probability distribution p0 with the transition matrix T estimated at lag time dlag*tau to time niter*dlag*tau
        If state_idx is provided, then the initial distribution is localised at the state of that idx.
        Otherwise, if stateidx is not provided, an arbitrary initial probability distribution can be given '''
    def chapman_kolmogorov_test(self,T,dlag,niter,state_idx=-1,p0=None):
        print "performing Chapman-Kolmogorov test...\n   propagating initial probability distribution %i times, lag time = %f" \
            % (niter,float(dlag)*self.tau)
        # check input parameters to function
        if state_idx<0 and p0 is None: raise IOError
        if state_idx>0: # using an initial probability distribution localised in the specified state
            assert state_idx<self.n_macrostates
            p0 = np.zeros(self.n_macrostates,dtype=float)
            p0[state_idx]=1.
        else: # using the provided initial probability distribution p0
            assert abs(np.sum(p0)-1.)<1.E-14
            for p0_val in p0: assert (p0_val>=0. and p0_val<=1.)
        # perform the Chapman-Kolmogorov test
        pt_arr = np.zeros((self.n_macrostates,niter+1),dtype=float) # array of time-dependent occupation probabilities p(t)
        t_vec = np.array([(float(i*dlag)*self.tau) for i in range(niter+1)]) # array of corresponding time values
        pt_arr[:,0] = p0 # initial distribution
        pt = p0
        for i in range(1,niter+1): # propagate the occupation probability distribution
            pt = np.dot(pt,T)
            assert abs(np.sum(pt)-1.)<1.E-12
            pt_arr[:,i] = pt
        return pt_arr, t_vec

    ''' plot the results of the Chapman-Kolmogorov test '''
    def plot_ck_test(self,pt_arr,t_vec,figfmt="pdf"):
        print "plotting results of Chapman-Kolmogorov test"
        plt.figure(figsize=(10.,7.)) # size in inches
        for i in range(self.n_macrostates): # plot p(t) for i-th state
            plt.plot(t_vec,pt_arr[i,:],linewidth=5)
        plt.xlabel("Time $t\ /\ \\tau$",fontsize=42)
        plt.ylabel("Occupation probability $p(t)$",fontsize=42)
        ax = plt.gca()
        ax.set_xlim([0,niter+1])
        ax.set_ylim([0.,1.])
        ax.tick_params(direction="out",labelsize=24)
        ax.set_xticks(t_vec)
        ax.set_yticks([0.+(float(i)*0.1) for i in range(11)])
        xticklabels=["$"+str(i)+"$" for i in range(niter+1)]
        yticklabels=["$"+str(0.+(float(i)*0.1))+"$" for i in range(11)]
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        plt.tight_layout()
        plt.savefig("ck_test."+figfmt,format=figfmt,bbox_inches="tight")
        plt.show()

    def correlation_func_test(self):
        pass

    ''' estimate a coarse-grained reversible DTMC with constrained stationary distribution, at discretised lag time
        (corresponding to the number of discrete time intervals in the discretised trajectories) dlag, by solving
        numerically for the maximum likelihood transition matrix '''
    def estimate_dtmc_mle(self,dlag=1):
        print "performing maximum-likelihood estimation of DTMC at lag time = %f" % (float(dlag)*self.tau)
        pyemma_mle_estimator = msm.MaximumLikelihoodMSM(lag=dlag,reversible=True, \
            statdist_constraint=self.pi_coarse,count_mode="sliding",maxiter=1000000,maxerr=1.E-08)
        pyemma_mle_msm_obj = pyemma_mle_estimator.estimate(self.dtrajs)
        for i in range(self.n_macrostates): assert abs(pyemma_mle_msm_obj.pi[i]-self.pi_coarse[i]) < 1.E-08
        return pyemma_mle_msm_obj.P, pyemma_mle_msm_obj.pi

    ''' estimate a coarse-grained reversible DTMC with constrained stationary distribution, at discretised lag time
        dlag, by using a Gibbs sampling Monte Carlo algorithm '''
    def estimate_dtmc_gibbs(self,dlag=1):
        print "performing Gibbs sampling of DTMC at lag time = %f" % (float(dlag)*self.tau)
        pyemma_gibbs_estimator = msm.BayesianMSM(lag=dlag,nsamples=100,nsteps=None,reversible=True, \
            statdist_constraint=self.pi_coarse,count_mode="effective",conf=0.95)
        pyemma_gibbs_msm_obj = pyemma_gibbs_estimator.estimate(self.dtrajs)
        for i in range(self.n_macrostates): assert abs(pyemma_gibbs_msm_obj.pi[i]-self.pi_coarse[i]) < 1.E-08
        return pyemma_gibbs_msm_obj.P, pyemma_gibbs_msm_obj.pi

    ''' estimate a coarse-grained reversible CTMC with constrained stationary distribution by maximum likelihood approach '''
    def estimate_ctmc_mle(self):
        print "performing maximum-likelihood estimation of CTMC"
        ctmc_obj = self.construct_ctmc_obj()
        tmtx_guess, pi_guess = self.estimate_dtmc_mle() # this DTMC is used to form an initial guess rate matrix
        ctmc_obj.fit_mle(tmtx_guess,pi_guess,guess_str="pseudo") # NB this is a custom function in a hacked version of MSMBuilder
        return ctmc_obj.ratemat_, ctmc_obj.populations_

    ''' construct an object representing a CTMC within the MSMBuilder package '''
    def construct_ctmc_obj(self):
#        ctmc_obj = ContinuousTimeMSM()
        ctmc_obj = CTMC_MLE_Obj()
        ctmc_obj.n_states_ = self.n_macrostates
        ctmc_obj.countsmat_ = self.get_counts_mtx()
        ctmc_obj.lag_time = self.tau
        return ctmc_obj

    ''' function to read kPS simulation data and convert these to 'discrete trajectories' at lag time tau.
        The kPS simulation data is contained in the files "walkers.x.y.dat", where x specifies the initial
        macrostate of the trajectory and y the ID of the trajectory in the set for this initial macrostate '''
    def readindtrajs(self,ntrajs_list,tau=None):
        print "reading in trajectories to discretised format"
        assert np.shape(ntrajs_list)[0]==self.n_macrostates
        if tau is None: tau = self.tau
        self.dtrajs = [[] for i in range(np.sum(ntrajs_list))] # record data in the current dtraj
        ntraj=0
        for i in range(self.n_macrostates):
            for j in range(ntrajs_list[i]):
                t=0. # next time for which the occupied macrostate is to be recorded
                walker_f =  open("walker."+str(i)+"."+str(j)+".dat","r")
                # read in all communities and times data
                trajdata = [(int(line.split()[1]),float(line.split()[2])) for line in walker_f.readlines()]
                walker_f.close()
                prev_comm = trajdata[0][0]
                for comm, time in trajdata:
                    while time >= t:
                        self.dtrajs[ntraj].append(prev_comm) # record data in the current dtraj
                        t += tau
                    prev_comm = comm
                ntraj += 1
        self.dtrajs = [np.array(dtraj,dtype=int) for dtraj in self.dtrajs]
        print "finished reading in %i trajectories in discretised representation" % ntraj

    ''' read a single-column file "fname" of the specified data type '''
    @staticmethod
    def read_single_col(fname,fmtfunc=int):
        p = subprocess.Popen(['wc','-l',fname],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        result, err = p.communicate()
        nlines = int(result.strip().split()[0])
        with open(fname) as data_f:
            data = [fmtfunc(next(data_f).split()[0]) for i in xrange(nlines)]
        return data

    ''' calculate the stationary probabilities for the macrostates as a simple sum over stationary probabilities of
        constitutent microstates '''
    def calc_pi_coarse(self):
        self.pi_coarse=np.full(shape=(self.n_macrostates),fill_value=float("-inf"),dtype=float)
        for i, pi_val in enumerate(self.pi):
            self.pi_coarse[self.comms[i]] = np.log(np.exp(self.pi_coarse[self.comms[i]])+np.exp(pi_val))
        assert abs(np.sum(np.array([np.exp(pi_val) for pi_val in self.pi_coarse]))-1.) < 1.E-10
        for i in range(self.n_macrostates): self.pi_coarse[i] = np.exp(self.pi_coarse[i])

    ''' construct the count matrix from discretised trajectories '''
    def get_counts_mtx(self):
        cmtx = np.zeros(shape=(self.n_macrostates,self.n_macrostates),dtype=int)
        for dtraj in self.dtrajs:
            prev_comm = dtraj[0]
            for comm in dtraj:
                cmtx[prev_comm,comm] += 1
                prev_comm = comm
        return cmtx


if __name__=="__main__":

    ### SET PARAMS FOR ESTIMATION ###
    n_macrostates = 11 # number of macrostates in the kPS simulation of the original Markov chain / number of microstates of the reduced Markov chain
    tau = 1.E+04 # lag time (or "tintvl") to read in trajectory data
    dlag = 2 # integer multiple of lag time at which trajectory data is interpreted (ie the lag time of the transition matrix is dlag*tau)
    ntrajs_list = Discotress_coarsegrainer.read_single_col("ntrajs.dat",int)
    ### SET PARAMS FOR TESTS AND PLOTS ###
    niter=15 # number of iterations in the Chapman-Kolmogorov test
    state_idx = 3 # initial probability distribution in Chapman-Kolmogorov test is localised in this state
    dlag_min = 1 # min dlag in testing for convergence of implied timescales
    dlag_max = 10 # max dlag in testing for convergence of implied timescales
    n_timescales = 8 # number of dominant implied timescales to compute

    ### RUN ###
    dcg1 = Discotress_coarsegrainer(n_macrostates,tau)
    dcg1.readindtrajs(ntrajs_list)
#    print dcg1.dtrajs
    T, pi = dcg1.estimate_dtmc_mle(dlag) # transition probability matrix and stationary distribution vector for DTMC from MLE
#    T, pi = dcg1.estimate_dtmc_gibbs() # transition probability matrix and stationary distribution vector for DTMC from Gibbs sampling
#    K, pi = dcg1.estimate_ctmc_mle() # transition rate matrix and stationary distribution vector for CTMC from MLE

    print "T:\n", T, "\n"
#    print "K:\n", K, "\n"
#    print "expm(tau*K):\n", expm(tau*K), "\n" # the MLE probability matrix and the exponential of the MLE rate matrix ought to be similar

    ### TEST AND PLOT ###
    # implied timescale test
    implt_arr, dlag_vec = dcg1.calc_implied_timescales(dlag_min,dlag_max,n_timescales=n_timescales)
    dcg1.implied_timescales_plot(implt_arr,dlag_vec)
    # Chapman-Kolmogorov test
    pt_arr, t_vec = dcg1.chapman_kolmogorov_test(T,dlag,niter,state_idx)
    dcg1.plot_ck_test(pt_arr,t_vec)
