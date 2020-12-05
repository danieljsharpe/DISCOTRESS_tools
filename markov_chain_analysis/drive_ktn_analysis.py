'''
Main Python script to drive estimation and analysis of a transition network.

Usage:
python drive_ktn_analysis.py <n_nodes> <n_edges> <n_comms> (<_ktn_id>)

Reading in a transition network requires at least 6x input data files:
    communities.dat, stat_prob.dat, edge_conns.dat, edge_weights.dat, nodes.A, nodes.B
additional optional files:
    edge_probs.dat, qf.dat/qb.dat

Daniel J. Sharpe
Nov 2019
'''

from __future__ import print_function
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

    Analyse_ktn.calc_tbranch(full_network)
    full_network.read_committors()

#    '''
    ### print KTN info ###
    print("\n\n")
    for node in full_network.nodelist:
        print("i:",node.node_id,"pi:",np.exp(node.pi),"q:",node.qf)
    print("\n\n")
    for edge in full_network.edgelist:
        print("from:    ",edge.from_node.node_id,"to    :",edge.to_node.node_id, \
              "\t\t\tk:\t","{:.6e}".format(np.exp(edge.k)),"    t:\t","{:.6e}".format(edge.t),"    fe:\t","{:.6e}".format(edge.fe))
#    '''


    ### calculate total reactive A<-B flux and the A-B transition state ensemble (TSE)
    totrflux, tse_edges = Analyse_ktn.get_isocommittor_cut(full_network,0.5,cumvals=False)
    print("\n\ntotal A<-B reactive flux:\t","{:.6e}".format(totrflux))
    print("\nTSE edges:")
    for edge, f, rel_f in tse_edges:
        print("  from: ",edge.from_node.node_id,"  to: ", edge.to_node.node_id,"    f:\t","{:.6e}".format(f),"    rel_f:\t","{:.6e}".format(rel_f))
    Analyse_ktn.write_flux_edgecosts(full_network)


    '''
    ### variational optimisation to find coarse-grained transition network ###
    coarse_ktn = full_network.construct_coarse_ktn()   
    analyser = Analyse_ktn()
    K_C_opt = analyser.varopt_simann(coarse_ktn,1000)
    '''

    '''
    ### calculate moments of path statistics ###
    manhart_morozov(full_network)
    '''
