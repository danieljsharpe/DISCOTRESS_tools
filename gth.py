'''
A plain Python implementation of the Grassmann-Taksar-Heyman (GTH) algorithm to compute the
stationary distribution of a Markov chain

Daniel J. Sharpe
May 2021
'''

from __future__ import print_function
import numpy as np


''' input: column-stochastic matrix T with N nodes
    output: stationary distribution vector pi '''
def gth_algo(T,N):
    print("\n\nrunning GTH algorithm...\n\n")
    # elimination phase
    for n in range(N-1,0,-1):
#        print("\neliminating node: ",n+1)
        Sn = np.sum([Tjn for j, Tjn in enumerate(T[:,n]) if j<n])
        T[n,:n] *= 1./Sn
        for i in range(n):
            for j in range(n):
                T[i,j] += T[i,n]*T[n,j]
    # trivial solution for one-node system
    pi = np.zeros(N,dtype=float)
    pi[0] = 1.
    # recursive phase
    mu = 1.
    for n in range(1,N):
        pi[n] = T[n,0] + np.sum([pi[j]*T[n,j] for j in range(1,n)])
        mu += pi[n]
    pi *= 1./mu # normalization
    return pi


if __name__=="__main__":

    T = np.load("transnmtx.pkl") # a row-stochastic matrix

    print("\n\nstochastic matrix:\n\n",T)

    T = T.T # transition matrix is now column-stochastic
    N = np.shape(T)[0] # number of nodes

    # dominant right eigenvector of an irreducible Markov chain is the stationary distribution
    evals, revecs = np.linalg.eig(T) # calculate right eigenvectors of the irreducible stochastic matrix
    revecs = np.array([revecs[:,i] for i in evals.argsort()[::-1]])
    evals = -np.sort(-evals)
    assert abs(evals[0]-1.)<1.E-08
    assert all([i>0 for i in revecs[0,:]])
    pi_evec = revecs[0,:]/np.sum(revecs[0,:])

    print("\neigenvalues:\n",evals)

    # checks on Markov chain
    assert isinstance(T,np.ndarray)
    assert np.shape(T)[1]==N
    assert len(np.shape(T))==2
    for i in range(N):
        assert abs(np.sum(T[:,i])-1.)<1.E-08


    print("\nstationary distribution from eigvec:\n",pi_evec)

    pi_gth = gth_algo(T,N) # stationary distribution
    print("\nstationary distribution from GTH:\n",pi_gth)
