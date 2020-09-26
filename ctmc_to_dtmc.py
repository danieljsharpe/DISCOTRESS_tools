'''
Python script to read files "ts_weights_ctmc.dat" and "ts_conns_ctmc.dat" parameterising a continuous-time Markov chain, calculate the matrix exponential at
specified lag time, and write the files "ts_weights_dtmc.dat" and "ts_conns_dtmc.dat" parameterising the discrete-time Markov chain
Also optionally reads the file "stat_prob.dat" and verifies that this stationary distribution is consistent with the estimated DTMC

Beware:
    matrix exponentiation may fail due to numerical issues for strongly metastable systems
    loss of numerical precision in transition probabilities from original transition rates

Daniel J. Sharpe
June 2020
'''

import numpy as np
from scipy.linalg import expm

class CTMC_To_DTMC(object):

    def __init__(self,tau,n_nodes):
        self.tau = tau
        self.n_nodes = n_nodes
        self.dtmc_mtx = None

    def estimate_dtmc(self):
        ctmc_rates = CTMC_To_DTMC.read_ts_weights()
        ctmc_conns = CTMC_To_DTMC.read_ts_conns()
        ctmc_mtx = self.construct_ctmc_mtx(ctmc_rates,ctmc_conns)
        self.dtmc_mtx = self.ctmc_to_dtmc(ctmc_mtx)

    def construct_ctmc_mtx(self,ctmc_rates,ctmc_conns):
        assert len(ctmc_rates)==len(ctmc_conns)
        ctmc_mtx = np.zeros((self.n_nodes,self.n_nodes),dtype=float)
        for i in range(len(ctmc_conns)):
            ctmc_mtx[ctmc_conns[i][0],ctmc_conns[i][1]] = np.exp(ctmc_rates[i][0])
            ctmc_mtx[ctmc_conns[i][1],ctmc_conns[i][0]] = np.exp(ctmc_rates[i][1])
        for i in range(self.n_nodes): # set diagonal entries so that rows sum to zero
            ctmc_mtx[i,i] = -np.sum(ctmc_mtx[:,i])
        return ctmc_mtx

    def ctmc_to_dtmc(self,ctmc_mtx):
        dtmc_mtx =  expm(ctmc_mtx*self.tau)
        for i in range(self.n_nodes): # check row-stochasticity
            assert abs(np.sum(dtmc_mtx[:,i])-1.) < 1.E-10
        return dtmc_mtx

    ''' read the file "stat_prob.dat" and check that this stationary probability distribution is consistent with the estimated DTMC '''
    def check_stat_prob(self):
        assert self.dtmc_mtx is not None
        pi = np.zeros(self.n_nodes,dtype=float)
        with open("stat_prob.dat","r") as pi_f:
            for i, line in enumerate(pi_f.readlines()): pi[i] = np.exp(float(line.split()[0]))
        assert abs(np.sum(pi)-1.)<1.E-10
        for i, elem in enumerate(np.dot(self.dtmc_mtx,pi)): # this dot product should return stationary probability vector
            if abs(elem-pi[i])>1.E-06: print i+1, abs(elem-pi[i])
#            assert abs(elem-pi[i])<1.E-06 # note relatively low tolerance here!

    def write_dtmc(self):
        assert self.dtmc_mtx is not None
        dtmc_probs = []
        dtmc_conns = []
        for i in range(self.n_nodes):
            for j in range(i+1,self.n_nodes): # note the diagonal entries can be inferred as the remainder from unity
                if self.dtmc_mtx[i,j]==0. and self.dtmc_mtx[j,i]==0.: continue # there is no connection (in either direction) for this pair of nodes
                dtmc_conns.append([i+1,j+1])
                dtmc_probs.append([self.dtmc_mtx[i,j],self.dtmc_mtx[j,i]])
        with open("ts_weights_dtmc.dat","w") as ts_wts_f:
            for prob_pair in dtmc_probs:
                ts_wts_f.write("%1.30f\n" % prob_pair[0])
                ts_wts_f.write("%1.30f\n" % prob_pair[1])
        with open("ts_conns_dtmc.dat","w") as ts_conns_f:
            for conn in dtmc_conns:
                ts_conns_f.write("%4i  %4i\n" % (conn[0],conn[1]))

    ''' read transition rates for CTMC, read into list as fwd/bwd pairs '''
    @staticmethod
    def read_ts_weights():
        ts_weights = []
        with open("ts_weights_ctmc.dat","r") as ts_wts_f:
            j=0
            for i, line in enumerate(ts_wts_f.readlines()):
                if i%2!=0:
                    ts_weights[j-1].append(float(line.split()[0]))
                    continue
                ts_weights.append([float(line.split()[0])])
                j+=1
        return ts_weights

    ''' read connections for CTMC '''
    @staticmethod
    def read_ts_conns():
        ts_conns = []
        with open("ts_conns_ctmc.dat","r") as ts_conns_f:
            for line in ts_conns_f:
                ts_conns.append([int(line.split()[0])-1,int(line.split()[1])-1])
                assert ts_conns[-1][0]!=ts_conns[-1][1]
        return ts_conns


if __name__=="__main__":

    ### CHOOSE PARAMS ###
    tau = 1.E+00 # lag time at which DTMC is calculated
    n_nodes = 16 # number of nodes in Markov chain

    ctmc_to_dtmc_obj = CTMC_To_DTMC(tau,n_nodes)
    ctmc_to_dtmc_obj.estimate_dtmc()
    ctmc_to_dtmc_obj.write_dtmc()
    ctmc_to_dtmc_obj.check_stat_prob()
    print ctmc_to_dtmc_obj.dtmc_mtx
