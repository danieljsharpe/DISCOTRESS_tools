'''
Algorithm of Manhart & Morozov (Phys Rev Lett 2013, J Chem Phys 2015, PNAS 2015) to calculate moments of the distributions of path statistics

Daniel J. Sharpe
'''

from __future__ import print_function
from ktn_structures import Node, Edge, Ktn, Coarse_ktn
from ktn_analysis_methods import Analyse_ktn
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from copy import copy
from math import factorial

''' class containing lists describing sparse matrix in CSR format and function to append an element to this list representation '''
class List_CSR(object):

    def __init__(self,dim1,dim2):
        self.row_vec = []
        self.col_vec = []
        self.data_vec = []
        self.nnz = 0 # number of non-zero elements
        self.dim1, self.dim2 = dim1, dim2

    def append_csr_elem(self,row_idx,col_idx,elem):
        if row_idx > self.dim1 or col_idx > self.dim2: raise AttributeError
        self.row_vec.append(row_idx)
        self.col_vec.append(col_idx)
        self.data_vec.append(elem)
        self.nnz += 1

    def return_csr_mtx(self):
        return csr_matrix((self.data_vec,(self.row_vec,self.col_vec)),shape=(self.dim1,self.dim2),dtype=float)

''' set up the initial transfer matrices,and transfer vectors needed by the algorithm of Manhart & Morozov '''
def setup_transfer_mtxs(ktn,nmax,Q,Qs_mtxs,Qh_mtxs,Theta_mtxs,debug):
    print(">>>>> setting up transfer matrices...")
    K_list = List_CSR(ktn.n_nodes*(nmax+1),ktn.n_nodes*(nmax+1)) # transfer matrix for dynamical activity
    G_list = List_CSR(ktn.n_nodes*(nmax+1),ktn.n_nodes*(nmax+1)) # transfer matrix for path action
    H_list = List_CSR(ktn.n_nodes*(nmax+1),ktn.n_nodes*(nmax+1)) # transfer matrix for path entropy
    F_list = List_CSR(nmax+1,ktn.n_nodes*(nmax+1)) # final sum matrix
    tau = np.zeros(shape=ktn.n_nodes*(nmax+1),dtype=float) # initial transfer vector for path times
    B_nodes = ktn.B.copy()
    B_node = B_nodes.pop() # node ID of the node in B (there should only be one node in B)
    assert not B_nodes
    tau[B_node.node_id-1] = 1. # initial distribution - there is only a single initial state
    # transfer matrices
    binomial_coeff = lambda n, k: float(factorial(n))/float((factorial(k)*factorial(n-k)))
    QTheta_mtxs = [Q.dot(Theta_mtxs[i]) for i in range(nmax+1)]
    # debugging
    if debug:
        print("\nQTheta_mtxs:\n", QTheta_mtxs)
        for i in range(nmax+1):
            print("\nQTheta", i, "\n", QTheta_mtxs[i].toarray())
    for i in range(nmax+1): # block rows
        for j in range(i+1): # block columns, matrices being built are lower triangular
            K_block_coo = QTheta_mtxs[i-j].tocoo()
            for row_idx, col_idx, elem in zip(K_block_coo.row,K_block_coo.col,K_block_coo.data):
                K_list.append_csr_elem((i*ktn.n_nodes)+row_idx,(j*ktn.n_nodes)+col_idx, \
                    binomial_coeff(i,i-j)*elem)
            G_block_coo = Qs_mtxs[i-j].tocoo()
            for row_idx, col_idx, elem in zip(G_block_coo.row,G_block_coo.col,G_block_coo.data):
                G_list.append_csr_elem((i*ktn.n_nodes)+row_idx,(j*ktn.n_nodes)+col_idx, \
                    binomial_coeff(i,i-j)*elem)
            H_block_coo = Qh_mtxs[i-j].tocoo()
            for row_idx, col_idx, elem in zip(H_block_coo.row,H_block_coo.col,H_block_coo.data):
                H_list.append_csr_elem((i*ktn.n_nodes)+row_idx,(j*ktn.n_nodes)+col_idx, \
                    binomial_coeff(i,i-j)*elem)
    # final sum matrix
    final_vec = sorted([node.node_id-1 for node in ktn.A]) # vector containing indicies of final nodes
    for i in range(nmax+1):
        for j in final_vec:
            F_list.append_csr_elem(i,(i*ktn.n_nodes)+j,1.)
    K = K_list.return_csr_mtx()
    G = G_list.return_csr_mtx()
    H = H_list.return_csr_mtx()
    F = F_list.return_csr_mtx()
    # more debugging
    if debug:
        print("\ntransfer matrix for time moments, K:\n", K)
        print("number of non-zero elems of K:", K.nnz)
    return K, G, H, F, tau

''' get the jump matrix (transition probability matrix when the elements are branching probabilities) in
    CSR sparse format, and similar matrices for calculating moments of other dynamical quantities '''
def get_jump_mtxs(ktn,nmax):
    Q_list = List_CSR(ktn.n_nodes,ktn.n_nodes) # jump matrix (used to calculate moments for dynamical activity)
    Qs_mtxs = [List_CSR(ktn.n_nodes,ktn.n_nodes) for i in range(nmax+1)] # matrices used to calculate moments for path action
    Qh_mtxs = [List_CSR(ktn.n_nodes,ktn.n_nodes) for i in range(nmax+1)] # matrices used to calculate moments for path entropy
    A_node_ids = [node.node_id for node in ktn.A]
    for edge in ktn.edgelist:
        if edge.deadts: continue
        if edge.from_node.node_id in A_node_ids: continue # final states are absorbing, corresponding column of jump matrix is all zeros
        Q_list.append_csr_elem(edge.to_node.node_id-1,edge.from_node.node_id-1,edge.t)
        for j, Qs_list in enumerate(Qs_mtxs):
            Qs_list.append_csr_elem(edge.to_node.node_id-1,edge.from_node.node_id-1,edge.t*((-(np.log(edge.t)))**j))
        for j, Qh_list in enumerate(Qh_mtxs):
            Qh_list.append_csr_elem(edge.to_node.node_id-1,edge.from_node.node_id-1, \
                edge.t*((-np.log((edge.t*np.exp(edge.from_node.k_esc))/(edge.rev_edge.t*np.exp(edge.to_node.k_esc))))**j))
    Q = Q_list.return_csr_mtx()
    # ensure that *columns* of jump matrix sum to 1 (note that elem Q_{ij} is the jump prob for the i<-j transition), except
    # for final (absorbing) nodes, where all elements must be zero
    for i in range(ktn.n_nodes):
        curr_col = Q.getcol(i).toarray()
        if i+1 in A_node_ids:
            assert all(elem==0. for elem in curr_col)
        else:
            assert abs(np.sum(curr_col)-1.) < 1.E-08
    for i in range(nmax+1):
        Qs_mtxs[i] = Qs_mtxs[i].return_csr_mtx()
        Qh_mtxs[i] = Qh_mtxs[i].return_csr_mtx()
    return Q, Qs_mtxs, Qh_mtxs

''' get the matrix with moments of the holding (waiting) time distributions for microstates along the diagonal '''
def get_holding_time_mtxs(ktn,nmax):
    Theta_mtxs = []
    for j in range(nmax+1):
        Theta_list = List_CSR(ktn.n_nodes,ktn.n_nodes)
        for i, node in enumerate(ktn.nodelist):
            if j==0: Theta_list.append_csr_elem(i,i,1.)
            elif j==1: Theta_list.append_csr_elem(i,i,1./np.exp(node.k_esc))
            else: Theta_list.append_csr_elem(i,i,factorial(j)*(1./np.exp(node.k_esc))**j)
        Theta = Theta_list.return_csr_mtx()
        Theta_mtxs.append(Theta)
    return Theta_mtxs

''' Main function for the algorithm of Manhart & Morozov.
    nmax is the number of moments of dsitributions of path statistics to calculate (default mean and standard deviation only)
    eps is a cutoff defining the convergence criteria '''
def manhart_morozov(ktn,nmax=2,eps=1.E-08,debug=False):
    print("\n>>>>> performing the algorithm of Manhart & Morozov to calculate the moments for distributions of path statistics\n")
    if not (isinstance(ktn,Ktn) or isinstance(ktn,Coarse_ktn)): raise AttributeError
    if len(ktn.B)!= 1: raise AttributeError # only allow a single source node
    Analyse_ktn.calc_tbranch(ktn) # get the transition matrix as the matrix of branching probabilities (jump matrix)
    Q, Qs_mtxs, Qh_mtxs = get_jump_mtxs(ktn,nmax)
    Theta_mtxs = get_holding_time_mtxs(ktn,nmax)
    K, G, H, F, tau = setup_transfer_mtxs(ktn,nmax,Q,Qs_mtxs,Qh_mtxs,Theta_mtxs,debug)
    # debugging - print all matrices
    if debug:
        print("\ninitial tau:\n", tau)
        print("\njump matrix:\n", Q.toarray())
        for i in range(nmax+1):
            print("\nholding time Theta matrix #", i)
            print(Theta_mtxs[i].toarray())
        print("\ntransfer matrix K:\n", K.toarray())
        print("\nfinal sum matrix F:\n", F.toarray())
    # all initial transfer vectors have the same form
    eta = copy(tau) # transfer vector for path action
    sigma = copy(tau) # transfer vector for path entropy
    tau_lprev, eta_lprev, sigma_lprev = copy(tau), copy(eta), copy(sigma)
    # main loop
    l=0
    t_L = np.zeros(shape=nmax+1,dtype=float) # accumulated path time probability distribution moments. Zeroth moment is path length
    s_L = np.zeros(shape=nmax+1,dtype=float) # accumulated path action probability distribution moments
    h_L = np.zeros(shape=nmax+1,dtype=float) # accumulated path entropy probability distribution moments
    path_moments_ts_f = open("path_moments_ts.dat","w")
    path_moments_ts_cum_f = open("path_moments_ts_cum.dat","w")
    path_moments_h_f = open("path_moments_h.dat","w")
    path_moments_h_cum_f = open("path_moments_h_cum.dat","w")
    path_moments_ts_f.write("# l / moments of time distribution / moments of action distribution\n")
    path_moments_ts_cum_f.write("# l / cum moments of time distribution / cum moments of action distribution\n")
    path_moments_h_f.write("# l / moments of entropy distribution\n")
    path_moments_h_cum_f.write("# l / cum moments of entropy distribution\n")
    while True:
        # calculate transfer vectors of the current iteration
        tau_l = K.dot(tau_lprev)
        eta_l = G.dot(eta_lprev)
        sigma_l = H.dot(sigma_lprev)
        # update cumulative transfer vectors 
        tau = tau+tau_l
        eta = eta+eta_l
        sigma = sigma+sigma_l
        tau_lprev, eta_lprev, sigma_lprev = copy(tau_l), copy(eta_l), copy(sigma_l)
        # collect sums for moments of path distributions for current iteration
        t_l = F.dot(tau_l)
        s_l = F.dot(eta_l)
        h_l = F.dot(sigma_l)
        # increment accumulated moment vectors
        t_L = t_L+t_l
        s_L = s_L+s_l
        h_L = h_L+h_l
        l+=1
        # write data to files
        path_moments_ts_f.write("%i    " % l)
        path_moments_ts_cum_f.write("%i    " % l)
        path_moments_h_f.write("%i    " % l)
        path_moments_h_cum_f.write("%i    " % l)
        for i in range(nmax+1):
            path_moments_ts_f.write("  %1.12f" % t_l[i])
            path_moments_ts_cum_f.write("  %1.12f" % t_L[i])
            path_moments_h_f.write("  %1.12f" % h_l[i])
            path_moments_h_cum_f.write("  %1.12f" % h_L[i])
        path_moments_ts_f.write("    ")
        path_moments_ts_cum_f.write("    ")
        for i in range(nmax+1):
            path_moments_ts_f.write("  %1.12f" % s_l[i])
            path_moments_ts_cum_f.write("  %1.12f" % s_L[i])
        path_moments_ts_f.write("\n")
        path_moments_ts_cum_f.write("\n")
        path_moments_h_f.write("\n")
        path_moments_h_cum_f.write("\n")
        # check convergence
        if 1.-t_L[0]<eps and t_l[nmax]>0. and t_l[nmax]/t_L[nmax]<eps:
            print(">>>>> path moments calculation converged after %i iterations" % l)
            break
    path_moments_ts_f.close()
    path_moments_ts_cum_f.close()
    path_moments_h_f.close()
    path_moments_h_cum_f.close()
    # write transfer (cumulative moment) vectors to files
    tau_f = open("tau_vec.dat","w")
    eta_f = open("eta_vec.dat","w")
    sigma_f = open("sigma_vec.dat","w")
    tau_f.write("# cumulative time moment vector over states, max path length: %i\n" % l)
    eta_f.write("# cumulative action moment vector over states, max path length: %i\n" % l)
    sigma_f.write("# cumulative entropy moment vector over states, max path length: %i\n" % l)
    for i in range(ktn.n_nodes):
        tau_f.write("%i" % (i+1))
        eta_f.write("%i" % (i+1))
        sigma_f.write("%i" % (i+1))
        for j in range(nmax+1):
            tau_f.write("  %1.12f" % tau[i+(j*ktn.n_nodes)])
            eta_f.write("  %1.12f" % eta[i+(j*ktn.n_nodes)])
            sigma_f.write("  %1.12f" % sigma[i+(j*ktn.n_nodes)])
        tau_f.write("\n")
        eta_f.write("\n")
        sigma_f.write("\n")
    tau_f.close()
    eta_f.close()
    sigma_f.close()
