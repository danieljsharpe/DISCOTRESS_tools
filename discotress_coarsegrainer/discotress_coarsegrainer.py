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
from scipy.linalg import eig
from math import floor
from math import ceil
from math import sqrt
import subprocess

''' class to read trajectory data output from an accelerated kMC simulation using DISCOTRESS,
    and perform estimation and validatation of a coarse-grained Markov chain '''
class Discotress_coarsegrainer(object):

    def __init__(self,n_macrostates,tau,trajtime):
        self.n_macrostates = n_macrostates
        self.tau = tau
        self.trajtime = trajtime
        self.pi = Discotress_coarsegrainer.read_single_col("stat_prob.dat",float)
        self.comms = Discotress_coarsegrainer.read_single_col("communities.dat",int)
        self.calc_pi_coarse()
        self.ntrajs_list = None
        self.dtrajs = None
        self.pltcolors=["royalblue","salmon","springgreen","gold","fuchsia","aqua","blueviolet", \
                        "seagreen","orange","mediumvioletred","najavowhite","papayawhip"]

    ''' calculate the implied timescales for a series of MLE transition probability matrices estimated at integer multiples of tau
        this information can be used to find the optimum lag time at which to estimate the transition matrices '''
    def calc_implied_timescales(self,dlag_min,dlag_max,dlag_intvl=1,n_timescales=None):
        assert (float((dlag_max-dlag_min))/float(dlag_intvl)).is_integer()
        niter=((dlag_max-dlag_min)/dlag_intvl)+1
        print "\n\ncalculating %i dominant implied timescales with base lag time: %f   number of iterations: %i   at intervals: %i" % \
              (n_timescales,self.tau,niter,dlag_intvl)
        if n_timescales is None: n_timescales=self.n_macrostates-1 # compute implied timescales for all states (note stationary implied timescale is ignored)
        else: assert n_timescales<self.n_macrostates # compute specified number of dominant implied timescales (ignoring stationary implied timescale)
        assert n_timescales<=12 # there are only 12 colours in the specified colours array
        implt_arr = np.zeros((n_timescales,niter),dtype=float)
        dlag_vec = np.array([dlag_min+(i*dlag_intvl) for i in range(niter)],dtype=int)
        for i, dlag in enumerate(dlag_vec):
            T, pi = self.estimate_dtmc_mle(dlag)
            T_eigvals, T_eigvecs = np.linalg.eig(T)
            T_eigvals = np.array(sorted(list(T_eigvals),reverse=True),dtype=float)
            for j in range(n_timescales):
                implt_arr[j,i] = (-1.*float(dlag)*tau)/np.log(T_eigvals[j+1])
        print "\narray of log_10 dominant implied timescales:\n", np.log10(implt_arr)
        return implt_arr, dlag_vec

    ''' plot the implied timescales to check for convergence '''
    def implied_timescales_plot(self,implt_arr,dlag_vec,figfmt="pdf"):
        print "\n\nplotting implied timescales"
        implt_arr=np.log10(implt_arr)
        plt.figure(figsize=(10.,7.)) # size in inches
        for i in range(np.shape(implt_arr)[0]):
            plt.plot(dlag_vec,implt_arr[i,:],linewidth=5,color=self.pltcolors[i],label="$t_{"+str(i+2)+"}$")
        plt.xlabel("Lag time $\\tau\ /\ t_\mathrm{intvl}$",fontsize=42)
        plt.ylabel("Implied timescale $\log_{10} t_k$",fontsize=42)
        plt.legend(loc="upper right",fontsize=24)
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
        Otherwise, if stateidx is not provided, an arbitrary initial probability distribution can be given.
        The time-dependent occupation probabilities of all the states in stateslist are recorded.
        Similar to the chapman_kolmogorov_test() function, except there the decay of probability initially localised in the current state of
        interest is considered in turn and only for that initially occupied state of the current iteration '''
    def propagate_ptdistribn(self,T,dlag,niter,intvl,stateslist,init_state_idx=-1,p0=None):
        print "\n\nperforming propagation of time-dependent occupation probability distribution...\n   " \
            "recording probability distribution %i times at intervals of %i, lag time = %f" % (niter,intvl,float(dlag)*self.tau)
        # check input parameters to function
        if init_state_idx<0 and p0 is None: raise IOError # if dont specify initial state must provide initial probability distribution p0
        if init_state_idx>0: # using an initial probability distribution localised in the specified state
            assert init_state_idx<self.n_macrostates
            p0 = np.zeros(self.n_macrostates,dtype=float)
            p0[init_state_idx]=1.
        else: # using the provided initial probability distribution p0
            assert abs(np.sum(p0)-1.)<1.E-14
            for p0_val in p0: assert (p0_val>=0. and p0_val<=1.)
        statesmask=[0]*self.n_macrostates
        for state_idx in stateslist: statesmask[state_idx]=1
        # perform the simulation on the coarse-grained Markov chain
        pt_arr = np.zeros((len(stateslist),niter+1),dtype=float) # array of time-dependent occupation probabilities p(t) for specified states
        t_vec = np.array([float(i*intvl*dlag)*self.tau for i in range(niter+1)]) # array of corresponding time values
        pt = p0 # set time-dependent distribution to initial distribution
        pt_arr[:,0] = [prob for k, prob in enumerate(p0) if statesmask[k]] # record initial distribution for specified states
        pt = p0.copy()
        for tint in range(1,(intvl*niter)+1): # propagate the occupation probability distribution
            pt = np.dot(pt,T)
            assert abs(np.sum(pt)-1.)<1.E-12
            if tint%intvl==0: # record occupation probabilities of specified states to array
                pt_arr[:,tint/intvl] = [prob for k, prob in enumerate(pt) if statesmask[k]]
        pt_traj_arr = None
        if init_state_idx>0: # using an initial probability distribution localised at particular state, so can compare to trajectories
            pt_traj_arr = self.read_counts_init_state(init_state_idx,stateslist,dlag,niter,intvl)
        print "\n  p(t) array for selected states determined from propagation using estimated coarse-grained Markov chain:\n", pt_arr
        print "\n  p(t) array for selected states determined from trajectory data:\n", pt_traj_arr
        return pt_arr, pt_traj_arr, t_vec

    ''' perform the Chapman-Kolmogorov test for the states listed in stateslist
        That is, propagate an initial probability distribution localised in the current state under consideration, using the transition matrix T
        estimated at lag time dlag*tau, up to time niter*dlag*tau.
        For each initially occupied state, only the time-dependent probability of that state is considered.
        The results are compared with the result from the trajectory data. '''
    def chapman_kolmogorov_test(self,T,dlag,niter,intvl,stateslist):
        print "\n\nperforming Chapman-Kolmogorov test...\n  propagating initial probability distributions %i times at intervals of %i, lag time = %f" \
            % (niter,intvl,float(dlag)*self.tau)
        pt_arr = np.zeros((len(stateslist),niter+1),dtype=float) # array of time-dependent occupation probabilities p(t) for specified states
        pt_traj_arr = np.zeros((len(stateslist),niter+1),dtype=float) # corresponding p(t) based on trajectory data, results of read_counts_init_state() func are copied into slices of this array
        t_vec = np.array([float(i*intvl*dlag)*self.tau for i in range(niter+1)])
        for j, state_idx in enumerate(stateslist): # consider initial occupation probability localised in each of the specified states in turn
            print "    performing CK test for state: %i" % (state_idx+1)
            p0 = np.zeros(self.n_macrostates,dtype=float)
            p0[state_idx]=1.; pt_arr[j,0]=1.
            pt = p0.copy() # set time-dependent distribution to initial distribution
            # propagate the occupation prob distribn with the coarse-grained prob matrix and record the prob of the initially occupied state
            for tint in range(1,(intvl*niter)+1):
                pt = np.dot(pt,T)
                assert abs(np.sum(pt)-1.)<1.E-12
                if tint%intvl==0: # record occupation probability of current initial state of iteration to array
                    pt_arr[j,tint/intvl] = pt[state_idx]
            # read in trajectory data and record the prob of the initially occupied state
            pt_traj_arr[j,:] = self.read_counts_init_state(state_idx,[state_idx],dlag,niter,intvl)[0,:].copy() # the call will return a (1xniter)-dimensional array
        print "\n  p(t) array for selected states determined from propagation using estimated coarse-grained Markov chain:\n", pt_arr
        print "\n  p(t) array for selected states determined from trajectory data:\n", pt_traj_arr
        return pt_arr, pt_traj_arr, t_vec
 
    ''' read from trajectories starting from specified initial state and record probabilities for chosen states at time intervals '''
    def read_counts_init_state(self,init_state_idx,stateslist,dlag,niter,intvl):
        pt_traj_arr = np.zeros((len(stateslist),niter+1),dtype=int) # p(t) from trajectory data, is initially an array of counts (int)
        statesmap=[-1]*self.n_macrostates # map of state_idx's in stateslist to indices of pt_traj_arr
        statemapidx=0
        for state_idx in stateslist:
            statesmap[state_idx]=statemapidx; statemapidx+=1
        max_ntraj=int(np.sum(self.ntrajs_list[:init_state_idx+1]))
        for dtraj in self.dtrajs[max_ntraj-self.ntrajs_list[init_state_idx]:max_ntraj]:
            # loop is over all trajs starting from initially occupied state under consideration
            assert np.shape(dtraj)[0]>=dlag*intvl*niter # otherwise there isnt enough trajectory data to do this many iterations
            for k, tidx in enumerate(range(0,(niter*dlag*intvl)+(dlag*intvl),dlag*intvl)):
                if statesmap[dtraj[tidx]]!=-1: pt_traj_arr[statesmap[dtraj[tidx]],k] += 1
        # convert counts in array computed from trajectory data to p(t)
        pt_traj_arr = pt_traj_arr.astype(float)
        for j, state_idx in enumerate(stateslist):
            pt_traj_arr[j,:] *= 1./self.ntrajs_list[init_state_idx]
            if init_state_idx==state_idx: assert abs(pt_traj_arr[j,0]-1.)<1.E-12 # should have only considered trajectories starting from relevant state
        return pt_traj_arr

    ''' plot the results of the Chapman-Kolmogorov test / propagation of the time-dependent occupation probability distribution for the states listed in stateslist '''
    def plot_ck_test(self,pt_arr,pt_traj_arr,t_vec,intvl,stateslist,figfmt="pdf"):
        print "\n\nplotting results for propagation of an occupation probability distribution"
        niter = np.shape(t_vec)[0]-1 # number of iterations of the C-K test
        plt.figure(figsize=(10.,7.)) # size in inches
        i=0
        for j, state_idx in enumerate(stateslist): # plot p(t) for i-th state; results from trajectory data drawn as continuous lines
            if pt_traj_arr is not None:
                plt.plot(t_vec,pt_traj_arr[j,:],linewidth=5,color=self.pltcolors[i],label=str(state_idx+1)) # note that the state labels in the plot are indexed from 1
            plt.scatter(t_vec,pt_arr[j,:],s=100,marker="o",color=self.pltcolors[i]) # results from coarse-grained Markov chain as dots
            i+=1
        plt.xlabel("Time $t\ /\ \\tau$",fontsize=42)
        plt.ylabel("Occupation probability $p(t)$",fontsize=42)
        plt.legend(loc="upper right",fontsize=24)
        ax = plt.gca()
        ax.set_xlim([0,niter+1])
        ax.set_ylim([0.,1.])
        ax.tick_params(direction="out",labelsize=24)
        ax.set_xticks(t_vec)
        ax.set_yticks([0.+(float(i)*0.1) for i in range(11)])
        xticklabels=["$"+str(i*intvl)+"$" for i in range(niter+1)]
        yticklabels=["$"+str(0.+(float(i)*0.1))+"$" for i in range(11)]
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        plt.tight_layout()
        plt.savefig("ck_test."+figfmt,format=figfmt,bbox_inches="tight")
        plt.show()

    ''' perform the correlation function test for specified auto- or cross-correlation functions [i,j]
        the probability distribution initially localised at the i-th state (computed from the discretised trajectories) is projected onto
        the j-th left eigenvector of the coarse-grained transition probability matrix T, estimated at lag time (dlag*self.tau)
        the correlation functions are analysed over the timeframe 0 to niter*intvl*dlag*self.tau '''
    def correlation_func_test(self,T,dlag,niter,intvl,corr_pairs):
        print "\n\nperforming correlation function tests for %i specified pairs of eigenvectors..." % len(corr_pairs)
        print "  checking correlation functions at %i different multiples, with interval %i, of lag time %f" % (niter,intvl,float(dlag)*self.tau)
        if self.dtrajs is None or self.ntrajs_list is None: raise IOError # not read in trajectory data yet!
        T_eigvals, T_reigvecs = eig(T,left=True,right=False) # note that this calculates the (unnormalised) *right* eigenvectors because T is transposed
        T_eigvals, T_leigvecs = eig(T,left=False,right=True) # note that this calculates the (unnormalised) *left* eigenvectors because T is transposed
        T_eigvals, T_reigvecs, T_leigvecs = Discotress_coarsegrainer.check_sort_eigs(T_eigvals,T_reigvecs,T_leigvecs,self.pi_coarse)
        print "    finished checking eigenvectors"
        corr_funcs_arr = np.zeros((len(corr_pairs),niter+1),dtype=float) # array of correlation function values
        t_vec = np.array([float(i*intvl*dlag)*self.tau for i in range(niter+1)]) # array of corresponding time values
        for j, corr_pair in enumerate(corr_pairs): # loop over specified pairs of eigenfunctions for which to calculate correlation functions
            print "\n    correlation function:  (%i, %i)" % (corr_pair[0]+1,corr_pair[1]+1)
            assert corr_pair[0]>0 and corr_pair[1]>0 # it doesn't make sense to look at correlation functions involving the stationary eigenmode, C_11(t) = 1 by definition
            for i, lagt in enumerate(t_vec):
                print "      multiple of lag time:", i*intvl
                val=0. # value of correlation function at this lag time
                for dtraj in self.dtrajs: # loop over all trajectories
                    k=0
                    while k+(i*intvl*dlag) < np.shape(dtraj)[0]: # process trajectory in shifting window until total trajectory time is met
                        val += T_leigvecs[corr_pair[1],dtraj[k+(i*intvl*dlag)]]*T_leigvecs[corr_pair[0],dtraj[k]]
                        k += 1
                    val *= 1./float(k)
                    corr_funcs_arr[j,i] += val # /float(np.shape(self.dtrajs[0]))
            corr_funcs_arr[j,:] *= 1./np.sum(self.ntrajs_list)
            if corr_pair[0]==corr_pair[1]: # expect single-exponential decay behaviour, otherwise values are strictly zero due to orthonormality of left eigenvectors
                print "\n  array of correlation function values determined from eigenspectrum of estimated coarse-grained Markov chain:\n", \
                    np.array([np.exp((1./(float(dlag)*tau))*np.log(T_eigvals[corr_pair[0]])*t) for t in t_vec])
            print "\n  array of correlation function values determined from trajectory data:\n", corr_funcs_arr[j,:]
        return corr_funcs_arr, t_vec, T_eigvals

    ''' plot the results of the correlation function test '''
    def plot_corrfunc_test(self,corr_funcs_arr,t_vec,T_eigvals,corr_pairs,tau,niter,intvl,figfmt="pdf"):
        print "\n\nplotting results of the correlation function test for %i pairs of dynamical eigenmodes" % len(corr_pairs)
        plt.figure(figsize=(10.,7.)) # size in inches
        if tau is not None: # indicates that the eigenvalues correspond to a DTMC, convert to CTMC eigenvalues
            T_eigvals = np.array([(1./tau)*np.log(T_eigval) for T_eigval in T_eigvals])
        for i in range(len(corr_pairs)):
            if corr_pairs[i][0]==corr_pairs[i][1]: # autocorrelation function should be a smooth exponential decay, plot this function
                tvals_eigval = np.linspace(t_vec[0],t_vec[-1],200) # time values for plotting the smooth exponential decay according to the eigenvalue
                plt.plot(tvals_eigval,[np.exp(T_eigvals[corr_pairs[i][0]]*t) for t in tvals_eigval], \
                     linewidth=3,color=self.pltcolors[i],zorder=10+i)
            plt.scatter(t_vec,corr_funcs_arr[i,:],s=100,marker="o",color=self.pltcolors[i], \
                     label="$C_{"+str(corr_pairs[i][0]+1)+str(corr_pairs[i][1]+1)+"}$",zorder=10+i) # note that the state labels in the plot are indexed from 1
        plt.xlabel("Time $t\ /\ \\tau$",fontsize=42)
        plt.ylabel("Correlation function $C_{lk}(t)$",fontsize=42)
        leg = plt.legend(loc="upper right",fontsize=24)
        leg.set_zorder(25) # legend goes on top
        ax=plt.gca()
        nxticks=niter+1
        assert abs((abs(t_vec[-1])+abs(t_vec[0]))%float(nxticks-1))<1.E-10 # enforce that time tick labels are integers
        nyticks=8; ymin=-0.2; ymax=1.2
        ytick_intvl=(abs(ymax)+abs(ymin))/float(nyticks-1)
        ax.set_xlim([t_vec[0],t_vec[-1]])
        ax.set_ylim([ymin,ymax])
        ax.tick_params(direction="out",labelsize=24)
        ax.set_xticks([round(x,0) for x in np.linspace(t_vec[0],t_vec[-1],nxticks)])
        ax.set_yticks([ymin+(float(i)*ytick_intvl) for i in range(nyticks)])
        xticklabels=["$"+str(i*intvl)+"$" for i in range(niter+1)]
        yticklabels=["$"+str((lambda x: x if x!=0. else abs(x))(round(ymin+(float(i)*ytick_intvl),1)))+"$" for i in range(nyticks)]
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        ax.axhline(y=0.,color="lightgray",linewidth=4,zorder=1)
        ax.axhline(y=1.,color="lightgray",linewidth=4,zorder=1)
        plt.tight_layout()
        plt.savefig("corrfunc_test."+figfmt,format=figfmt,bbox_inches="tight")
        plt.show()

    ''' estimate a coarse-grained reversible DTMC with constrained stationary distribution, at discretised lag time
        (corresponding to the number of discrete time intervals in the discretised trajectories) dlag, by solving
        numerically for the maximum likelihood transition matrix '''
    def estimate_dtmc_mle(self,dlag=1):
        print "\nperforming maximum-likelihood estimation of DTMC at lag time = %f" % (float(dlag)*self.tau)
        pyemma_mle_estimator = msm.MaximumLikelihoodMSM(lag=dlag,reversible=True, \
            statdist_constraint=self.pi_coarse,count_mode="sliding",mincount_connectivity=0,maxiter=1000000,maxerr=1.E-08)
        pyemma_mle_msm_obj = pyemma_mle_estimator.estimate(self.dtrajs)
        Discotress_coarsegrainer.check_fully_connected(pyemma_mle_msm_obj.count_matrix_full)
        assert pyemma_mle_msm_obj.nstates==self.n_macrostates # MSM object can have fewer states if trajectories dont yield fully connected count matrix
        for i in range(self.n_macrostates): assert abs(pyemma_mle_msm_obj.pi[i]-self.pi_coarse[i]) < 1.E-08
        Discotress_coarsegrainer.check_row_stochasticity(pyemma_mle_msm_obj.P,pyemma_mle_msm_obj.pi)
        return pyemma_mle_msm_obj.P, pyemma_mle_msm_obj.pi

    ''' estimate a coarse-grained reversible DTMC with constrained stationary distribution, at discretised lag time
        dlag, by using a Gibbs sampling Monte Carlo algorithm '''
    def estimate_dtmc_gibbs(self,dlag=1):
        print "\nperforming Gibbs sampling of DTMC at lag time = %f" % (float(dlag)*self.tau)
        pyemma_gibbs_estimator = msm.BayesianMSM(lag=dlag,nsamples=100,nsteps=None,reversible=True, \
            statdist_constraint=self.pi_coarse,count_mode="effective",mincount_connectivity=0,conf=0.95)
        pyemma_gibbs_msm_obj = pyemma_gibbs_estimator.estimate(self.dtrajs)
        Discotress_coarsegrainer.check_fully_connected(pyemma_gibbs_estimator.count_matrix_full)
        assert pyemma_mle_msm_obj.nstates==self.n_macrostates
        for i in range(self.n_macrostates): assert abs(pyemma_gibbs_msm_obj.pi[i]-self.pi_coarse[i]) < 1.E-08
        Discotress_coarsegrainer.check_row_stochasticity(pyemma_gibbs_msm_obj.P,pyemma_gibbs_msm_obj.pi)
        return pyemma_gibbs_msm_obj.P, pyemma_gibbs_msm_obj.pi

    ''' estimate a coarse-grained reversible CTMC with constrained stationary distribution by maximum likelihood approach '''
    def estimate_ctmc_mle(self):
        print "\nperforming maximum-likelihood estimation of CTMC"
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
        assert np.shape(ntrajs_list)[0]==self.n_macrostates
        if tau is None: tau = self.tau
        assert (self.trajtime/tau).is_integer()
        print "\nreading in trajectories up to total time: %f    into discretised format at lag time: %f" % (self.trajtime,tau)
        self.ntrajs_list = ntrajs_list
        dummy_arr = np.zeros(int(self.trajtime/tau)+1,dtype=int)
        self.dtrajs = [dummy_arr.copy() for i in range(np.sum(self.ntrajs_list))] # array of discretised trajectories, must be list of int-ndarray's
        ntraj=0 # current trajectory being read in
        for i in range(self.n_macrostates): # loop over sets of trajectories starting from each macrostate
            for j in range(ntrajs_list[i]): # loop over individual trajectories starting from this macrostate
                walker_f =  open("walker."+str(i)+"."+str(j)+".dat","r")
                # read in all communities and times data
                trajdata = [(int(line.split()[1]),float(line.split()[3])) for line in walker_f.readlines()]
                walker_f.close()
                prev_comm = trajdata[0][0]
                t=0. # next time for which the occupied macrostate is to be recorded
                k=0
                for comm, time in trajdata:
                    while time >= t and t <= self.trajtime:
                        self.dtrajs[ntraj][k] = prev_comm # record data in the current dtraj
                        t += tau
                        k += 1
                    prev_comm = comm
                    if time > self.trajtime: break
                assert k==int(self.trajtime/tau)+1 # check that correct number of entries have been recorded in the dtrajs array
                ntraj += 1
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

    ''' check that a transition matrix is row-stochastic and that the stationary probability distribution is stationary '''
    @staticmethod
    def check_row_stochasticity(T,pi):
        for i in range(np.shape(T)[0]): assert abs(np.sum(T[i,:])-1.)<1.E-08
        for i, prob in enumerate(np.dot(pi,T)): assert abs(prob-pi[i])<1.E-08

    ''' check that the square matrix C_mtx is fully connected (check Fiedler value, ie smallest value of Laplacian matrix, is positive) '''
    @staticmethod
    def check_fully_connected(C_mtx):
        A_mtx = np.zeros((np.shape(C_mtx)[0],np.shape(C_mtx)[0]),dtype=float) # adjacency matrix (with zeroes along diagonal)
        for i in range(np.shape(C_mtx)[0]):
            for j in range(np.shape(C_mtx)[0]):
                if C_mtx[i,j]: A_mtx[i,j]=1.
        D_vec = np.array([np.sum(A_mtx[:,i]) for i in range(np.shape(A_mtx)[0])]) # vector of node degrees in matrix
        L_eigvals, L_eigvecs = np.linalg.eig(np.diag(D_vec)-A_mtx) # eigenspectrum of Laplacian matrix
        assert sorted(L_eigvals,reverse=True)[0]>0.

    ''' sort eigenvalues, then sort right and left eigenvectors in order of corresponding eigenvalues, and renormalise eigenvectors
        the normalised eigenvectors are checked (eg dominant right eigenvector is stationary distribution, left and right eigenvectors
        satisfy normalisation conditions)
        note also that the left and right eigenvector arrays become transposed '''
    @staticmethod
    def check_sort_eigs(eigvals,reigvecs,leigvecs,stat_distribn):
        n_states = np.shape(reigvecs)[0]
        reigvecs = np.array([eigvec for _, eigvec in sorted(zip(list(eigvals),list(np.transpose(reigvecs))), \
            key=lambda pair: pair[0], reverse=True)],dtype=float)
        leigvecs = np.array([eigvec for _, eigvec in sorted(zip(list(eigvals),list(np.transpose(leigvecs))), \
            key=lambda pair: pair[0], reverse=True)],dtype=float)
        eigvals = np.array(sorted(list(eigvals),reverse=True),dtype=float)
        # compute normalisation factors
        tmp_arr_right = np.zeros((n_states,n_states),dtype=float) # elements required for normalisation of right eigenvectors
        tmp_arr_left = np.zeros((n_states,n_states),dtype=float) # elements required for normalisation of left eigenvectors
        for i in range(n_states):
            for j in range(n_states):
                for k in range(n_states):
                    tmp_arr_right[i,j] += reigvecs[i,k]*reigvecs[j,k]/stat_distribn[k]
                    tmp_arr_left[i,j] += leigvecs[i,k]*leigvecs[j,k]*stat_distribn[k]
        # normalise
        for i in range(n_states):
            reigvecs[i,:] *= 1./sqrt(tmp_arr_right[i,i])
            leigvecs[i,:] *= 1./sqrt(tmp_arr_left[i,i])
        # various checks for normalisation and orthogonality of left and right eigenvectors
        for i in range(n_states): assert abs(reigvecs[0,i]-stat_distribn[i])<1.E-08
        for i in range(n_states):
            assert abs(np.sum(reigvecs[i,:])-(lambda i: 1. if i==0 else 0.)(i))<1.E-10
            assert abs(np.dot(stat_distribn,leigvecs[i,:])-(lambda i: 1. if i==0 else 0.)(i))<1.E-10
            for j in range(n_states):
                if i==j: continue
                assert abs(tmp_arr_right[i,j])<1.E-10
                assert abs(tmp_arr_left[i,j])<1.E-10
        return eigvals, reigvecs, leigvecs

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
    trajtime = 1.E+07 # read in data up until this time (which must be met) for *all* trajectories (note that trajectories may exceed this length in time)
    dlag = 2 # integer multiple of lag time at which trajectory data is interpreted (ie the lag time of the transition matrix is dlag*tau)
    ntrajs_list = Discotress_coarsegrainer.read_single_col("ntrajs.dat",int)
    ### SET PARAMS FOR TESTS, ANALYSIS, AND PLOTS ###
    # implied timescale convergence test
    dlag_min = 1 # min dlag in testing for convergence of implied timescales
    dlag_max = 21 # max dlag in testing for convergence of implied timescales
    dlag_intvl = 2 # interval (multiple of base lag time) in testing for convergence of implied timescales
    n_timescales = 8 # number of dominant implied timescales to compute
    # Chapman-Kolmogorov test / propagation of occupation probability distribution
    niter_ck = 15 # number of iterations in the Chapman-Kolmogorov test / propagation
    tintvl_ck = 2 # time interval (multiple of lag time) for recording p(t) in Chapman-Kolmogorov test / propagation
    state_idx = 3 # initial probability distribution in propagation of p(t) is localised in this state
    p0 = None # alternatively to providing a state_idx for the propagation of p(t), can specify an arbitrary initial probability distribution
    stateslist = [0,3,5,8] # list of states to plot the occupation probability distributions for in the CK test (or p(t) propagation) (indexed from zero)
    # correlation function test
    niter_cf = 15 # number of iterations in the correlation function test
    tintvl_cf = 4 # interval of lag times for correlation function test
    # pairs of dominant eigenvectors (indexed from 0) [i,j] for which to compute correlation functions. i=j and i=/=j are auto- and cross-correlation funcs, respectively
    corr_pairs = [[2,2],[4,4],[2,3]]


    ### RUN ###
    # set up object and read in trajectory data
    dcg1 = Discotress_coarsegrainer(n_macrostates,tau,trajtime)
    dcg1.readindtrajs(ntrajs_list)

    # construct Markov chain
    T, pi = dcg1.estimate_dtmc_mle(dlag) # transition probability matrix and stationary distribution vector for DTMC from MLE
#    T, pi = dcg1.estimate_dtmc_gibbs() # transition probability matrix and stationary distribution vector for DTMC from Gibbs sampling
#    K, pi = dcg1.estimate_ctmc_mle() # transition rate matrix and stationary distribution vector for CTMC from MLE

    print "T:\n", T, "\n"
#    print "K:\n", K, "\n"
#    print "expm(tau*K):\n", expm(tau*K), "\n" # the MLE probability matrix and the exponential of the MLE rate matrix ought to be similar


    ### TESTS AND PLOTS ###
    # implied timescale test
    implt_arr, dlag_vec = dcg1.calc_implied_timescales(dlag_min,dlag_max,dlag_intvl,n_timescales=n_timescales)
    dcg1.implied_timescales_plot(implt_arr,dlag_vec)

    # propagate time-dependent probability distribution p(t) for several states given initial condition
    pt_arr, pt_traj_arr, pt_t_vec = dcg1.propagate_ptdistribn(T,dlag,niter_ck,tintvl_ck,stateslist,state_idx)
    dcg1.plot_ck_test(pt_arr,pt_traj_arr,pt_t_vec,tintvl_ck,stateslist)

    # Chapman-Kolmogorov test
    ck_arr, ck_traj_arr, ck_t_vec = dcg1.chapman_kolmogorov_test(T,dlag,niter_ck,tintvl_ck,stateslist)
    dcg1.plot_ck_test(ck_arr,ck_traj_arr,ck_t_vec,tintvl_ck,stateslist)

    # correlation function test
    corr_funcs_arr, t_vec, T_eigvals = dcg1.correlation_func_test(T,dlag,niter_cf,tintvl_cf,corr_pairs)
    dcg1.plot_corrfunc_test(corr_funcs_arr,t_vec,T_eigvals,corr_pairs,tau*float(dlag),niter_cf,tintvl_cf)
