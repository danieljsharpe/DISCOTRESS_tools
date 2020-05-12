'''
    A Python object representing a CTMC and containing functions required for solution of maximum likelihood estimation (MLE) constrained
    optimisation problem to estimate a CTMC

    Daniel J. Sharpe
'''

from __future__ import print_function, division
import numpy as np
import scipy.linalg
import scipy.optimize
import pyximport; pyximport.install()
import _ratematrix # Cython code to calculate log likelihood of reversible CTMC

class CTMC_MLE_Obj(object):

    ''' NB variable names are chosen to be consistent with the ContinuousTimeMSM object of the MSMBuilder package '''
    def __init__(self):
        self.n_states_ = None
        self.countsmat_ = None
        self.ratemat_ = None
        self.populations_ = None
        self.lag_time = None

    @property
    def n_states_(self):
        return self.__n_states_

    @n_states_.setter
    def n_states_(self,n_states_):
        self.__n_states_ = n_states_

    @property
    def countsmat_(self):
        return self.__countsmat_

    @countsmat_.setter
    def countsmat_(self,countsmat_):
        self.__countsmat_ = countsmat_

    @property
    def lag_time(self):
        return self.__lag_time

    @lag_time.setter
    def lag_time(self,lag_time):
        self.__lag_time  = lag_time

    def fit_mle(self,tmtx_guess,pi_guess,guess_str="pseudo"):
        print("performing maximum likelihood estimation of a CTMC")
        # get an initial guess for the transition rate matrix
        if guess_str=="pseudo": # initial rate matrix is the first order Taylor expansion of the matrix exponential
            kmtx_init = (tmtx_guess-np.eye(self.n_states_))/self.lag_time
        elif guess_str=="log": # initial rate matrix is the matrix log of the transition probability matrix
            kmtmx_init = np.real(scipy.linalg.logm(tmtx_guess))/self.lag_time
        else:
            raise RuntimeError
        smtx = np.multiply(np.sqrt(np.outer(pi_guess,1./pi_guess)),kmtx_init)
        sflat = np.maximum(smtx[np.triu_indices_from(self.countsmat_,k=1)],0)
        theta0 = np.concatenate((sflat,np.log(pi_guess))).astype("float") # theta contains all elements needed to construct the reversible CTMC
        loglikelihoods = []
        countsmat = self.countsmat_.astype(float)
        lag_time = self.lag_time
        def objective(theta):
            f, g = _ratematrix.loglikelihood(theta,countsmat,lag_time)
            if not np.isfinite(f): f = np.nan
            loglikelihoods.append(f)
            return -f, -g
        # this bound prevents the stationary probability for any state from getting too small (helps numerical stability)
        bounds = ([(0,None)]*int((self.n_states_*(self.n_states_-1)/2))) + ([(-20,None)]*self.n_states_)
        result = scipy.optimize.minimize(fun=objective,x0=theta0,method="L-BFGS-B",jac=True,bounds=bounds)
        theta_final = result.x
        # check that the stationary distribution constraint is obeyed (pi_guess should already have been checked against the specified distribution)
#        assert [ check pi_guess vs relevant (ie final n_states_) elemes of theta_final ]
        # construct the rate matrix from the elements of theta_final
        K = np.zeros((self.n_states_, self.n_states_))
        _ratematrix.build_ratemat(result.x, self.n_states_, K, which='K')
        self.ratemat_ = K
        return self.ratemat_, loglikelihoods # return rate matrix and log likelihoods

