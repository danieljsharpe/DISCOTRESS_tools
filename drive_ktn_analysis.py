'''
Main Python script to drive estimation and analysis of a transition network.

Usage:
python coarse_ktn_analysis.py <n_nodes> <n_edges> <n_comms> <_ktn_id>

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
    ktn_id = str(argv[4])
    assert ktn_id[0]=="_" # leading character of ktn_id should be underscore
    full_network = Ktn(n_nodes,n_edges,n_comms,ktn_id)
    comms, conns, pi, k, t, node_ens, ts_ens = Ktn.read_ktn_info(n_nodes,n_edges,ktn_id)
    full_network.construct_ktn(comms,conns,pi,k,t,node_ens,ts_ens)

    full_network.read_committors()
    full_network.renormalise_mr()
    Analyse_ktn.isocommittor_cut_analysis(full_network,ncuts=3)
    full_network.print_gephi_fmt()

    '''
    ### SIMPLE TESTS ###
    mynode1 = Node(1)
    mynode1.node_attribs = [-0.2,1,0.45]
    mynode1.tpt_vals = [0.3,0.7]
    mynode2 = Node(6)
    mynode2.node_attribs = [-0.4,2,0.30]
    myedge1 = Edge(5,5)
    myedge1.to_from_nodes = ([mynode1,mynode2],True)
    mynode1.node_id = 2
    print "edge #1 to/from:", myedge1.to_from_nodes
    print "ID of first IN edge of node 1:", mynode1.edgelist_in[0].edge_id
    print repr(mynode1), "\n", str(mynode1)
    del mynode1.node_attribs
    print "forward committor for node 1 has now been deleted. qf:", mynode1.qf

    mynode3 = Node(3)
    mynode3.node_attribs = [-0.5,4,0.25]
    myedge2 = Edge(8,8)
    myedge2.to_from_nodes = ([mynode1,mynode3],True)
    del myedge1.to_from_nodes
    print "new ID of first IN edge of node 1:", mynode1.edgelist_in[0].edge_id
    '''

    '''
    ### TEST KTN ###
    print "\n\n\n"
    print "nbrlist for node 333: ", full_network.nodelist[332].nbrlist
    print "edgelist_in for node 333: ", full_network.nodelist[332].edgelist_in
    print "edgelist_out for node 333: ", full_network.nodelist[332].edgelist_out
    print "stationary probabilities of communities:\n", [np.exp(x) for x in full_network.comm_pi_vec]
    print "out edges for node 1:", len(full_network.nodelist[0].edgelist_out)
    '''
    '''
    ### test eval/evec calculations on transition network ###
    print "\ndominant eigenvalues of transition rate matrix:"
    analyser = Analyse_ktn()
    K_sp = Analyse_ktn.setup_sp_k_mtx(full_network)
    K_sp_eigs, K_sp_evecs = Analyse_ktn.calc_eig_iram(K_sp,7)
    print K_sp_eigs

    tau = 1.E+1
    print "\n eigenvalues of transition matrix, lag time:", tau
    print [Analyse_ktn.eigs_K_to_T(g,tau) for g in K_sp_eigs]

    print "\n characteristic timescales of full matrix:"
    print [1./eig for eig in K_sp_eigs]
    '''

    '''
    ### SET UP COARSE NETWORK ###
    print "\nforming the coarse matrix:"
    coarse_ktn = full_network.construct_coarse_ktn()
    '''

    '''
    ### TEST COARSE NETWORK ###
    print "endpoint macrostates:"
    print "A:", coarse_ktn.A, "B:", coarse_ktn.B

    print "no. of nodes:", coarse_ktn.n_nodes
    print "nodelist:", coarse_ktn.nodelist
    print "edgelist:", coarse_ktn.edgelist
    print "nbrlist for comm 2: ", coarse_ktn.nodelist[2].nbrlist # need to except dead TSs
    print "edgelist_in for comm 2: ", coarse_ktn.nodelist[2].edgelist_in # ditto
    print "edgelist_out for comm 2: ", coarse_ktn.nodelist[2].edgelist_out # ditto

    print "stationary probabilities of coarse nodes:\n", [np.exp(x.pi) for x in coarse_ktn.nodelist]

    print "\neigenvalues of coarse matrix (sparse):"
    K_C_sp = Analyse_ktn.setup_sp_k_mtx(coarse_ktn)
    K_C_sp_eigs, K_C_sp_evecs = Analyse_ktn.calc_eig_iram(K_C_sp,3)
    print K_C_sp_eigs
    print "\neigenvalues of coarse matrix (not sparse):"
    K_C = Analyse_ktn.setup_k_mtx(coarse_ktn)
    K_C_eigs, K_C_evecs = Analyse_ktn.calc_eig_all(K_C)
    print K_C_eigs
    print "\ncoarse transition rate matrix:"
    print K_C
    '''

    # calculate committor functions by SLSQP constrained linear optimisation
#    Analyse_ktn.calc_committors(full_network,method="linopt")
    # OR read committor functions from files
#    full_network.read_committors()

    '''
    ### ESTIMATE COARSE NETWORK BY VARIATIONAL OPTIMISATION ###
    print "\n doing variational optimisation of coarse rate matrix:"
    try: analyser
    except NameError: analyser = Analyse_ktn()
    K_C_opt = analyser.varopt_simann(coarse_ktn,5000)
    print "\neigenvalues of coarse matrix (after var opt procedure):"
    K_C_opt_eigs, K_C_opt_evecs = Analyse_ktn.calc_eig_all(K_C_opt)
    print K_C_opt_eigs
    print "\n characteristic timescales of coarse matrix:"
    print [1./eig for eig in K_C_opt_eigs]
    print "stationary probabilities of coarse nodes:\n", [np.exp(x.pi) for x in coarse_ktn.nodelist]
    '''

    '''
    ### ISOCOMMITTOR CUT ANALYSIS ###
    Analyse_ktn.isocommittor_cut_analysis(full_network,3,writedata=False)
    Analyse_ktn.calc_tlin(full_network,2.E-1)
    '''

    '''
    ### RELATIVE ENTROPY ANALYSIS ###
    # create a second network and calculate relative entropy metrics between the two networks
    network_2 = Ktn(n_nodes,n_edges,n_comms)
    comms, conns, pi, k, t, node_ens, ts_ens = Ktn.read_ktn_info(n_nodes,n_edges,ktn_id="_3h_t01")
    network_2.construct_ktn(comms,conns,pi,k,t,node_ens,ts_ens)
    Analyse_ktn.calc_tlin(network_2,2.E-1)
#    Analyse_ktn.calc_surprisal(full_network,network_2)
    '''

    '''
    ### TOY NETWORK ###
    toy_network = Ktn(4,4,1)
    toy_network2 = Ktn(4,4,1)
    comms, conns, pi, k, t, node_ens, ts_ens = Ktn.read_ktn_info(4,4,ktn_id="_toy")
    print "checking sum over stat_prob:", sum([np.exp(pi_i) for pi_i in pi])
    toy_network.construct_ktn(comms,conns,pi,k,t,node_ens,ts_ens)
    comms, conns, pi, k, t, node_ens, ts_ens = Ktn.read_ktn_info(4,4,ktn_id="_toy_t2")
    toy_network2.construct_ktn(comms,conns,pi,k,t,node_ens,ts_ens)
    Analyse_ktn.calc_surprisal(toy_network,toy_network2)
    '''

    '''
    ### get eigenvectors and dump information to files ###
    coarse_ktn.get_eigvecs()
    coarse_ktn.write_nodes_info()
    coarse_ktn.write_edges_info()
    '''

    ### write transition network output in Gephi format '''
    # full_network.print_gephi_fmt(evec_idx=0)

    '''
    ### CALCULATE MOMENTS OF PATH STATISTICS ###
    manhart_morozov(full_network)
    '''
