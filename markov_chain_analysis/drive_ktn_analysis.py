'''
Main Python script to drive estimation and analysis of a transition network.

Usage:
python drive_ktn_analysis.py <n_nodes> <n_edges> <n_comms> (<_ktn_id>)

Reading in a transition network requires at least 6x input data files:
communities.dat, stat_prob.dat, ts_conns.dat, ts_weights.dat, min.A, min.B

Daniel J. Sharpe
Nov 2019
'''

from ktn_structures import Node, Edge, Ktn, Coarse_ktn
from ktn_analysis_methods import Analyse_ktn
from transfer_matrices import manhart_morozov
import numpy as np
from sys import argv
from sys import setrecursionlimit

if __name__=="__main__":

    setrecursionlimit(100000)

    ### MAIN ###
    n_nodes = int(argv[1])
    n_edges = int(argv[2])
    n_comms = int(argv[3])
    try:
        ktn_id = str(argv[4])
        assert ktn_id[0]=="_" # leading character of ktn_id should be underscore
    except IndexError:
        ktn_id = ""
    full_network = Ktn(n_nodes,n_edges,n_comms,ktn_id)
    comms, conns, pi, k, t, node_ens, ts_ens = Ktn.read_ktn_info(n_nodes,n_edges,ktn_id)
    full_network.construct_ktn(comms,conns,pi,k,t,node_ens,ts_ens)

#    Analyse_ktn.calc_tlin(full_network,1.E+01)
    Analyse_ktn.calc_tsameeig(full_network,0.8,1.)
    Analyse_ktn.dump_tprobs(full_network)

    '''
    # variational optimisation to find coarse-grained transition network
    coarse_ktn = full_network.construct_coarse_ktn()   
    analyser = Analyse_ktn()
    K_C_opt = analyser.varopt_simann(coarse_ktn,1000)
    '''

    '''
    ### CALCULATE MOMENTS OF PATH STATISTICS ###
    manhart_morozov(full_network)
    '''
