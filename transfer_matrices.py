'''
Algorithm of Manhart & Morozov (Phys Rev Lett 2013, J Chem Phys 2015, PNAS 2015) to calculate moments of the distributions of path statistics

Daniel J. Sharpe
'''

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
def setup_transfer_mtxs(ktn,nmax,Q,Qs_mtxs,Theta_mtxs):
    print ">>>>> setting up transfer matrices..."
    K_list = List_CSR(ktn.n_nodes*(nmax+1),ktn.n_nodes*(nmax+1)) # transfer matrix for dynamical activity
    G_list = List_CSR(ktn.n_nodes*(nmax+1),ktn.n_nodes*(nmax+1)) # transfer matrix for path action
    F_list = List_CSR(nmax+1,ktn.n_nodes*(nmax+1)) # final sum matrix
    tau = np.zeros(shape=ktn.n_nodes*(nmax+1),dtype=float) # initial transfer vector for path times
    eta = np.zeros(shape=ktn.n_nodes*(nmax+1),dtype=float) # initial transfer vector for path action
    delS = np.zeros(shape=ktn.n_nodes*(nmax+1),dtype=float)
    B_nodes = ktn.B.copy()
    B_node = B_nodes.pop() # node ID of the node in B (there should only be one node in B)
    assert not B_nodes
    tau[B_node.node_id-1] = 1. # initial distribution - there is only a single initial state
    eta[B_node.node_id-1] = 1.
    # transfer matrices
    binomial_coeff = lambda n, k: float(factorial(n))/float((factorial(k)*factorial(n-k)))
    QTheta_mtxs = [Q.dot(Theta_mtxs[i]) for i in range(nmax+1)]

    print "\nQTheta_mtxs:\n", QTheta_mtxs
    for i in range(nmax+1):
        print "\nQTheta", i, "\n", QTheta_mtxs[i].toarray()

    for i in range(nmax+1): # block rows
        for j in range(i+1): # block columns, matrices being built are lower triangular
#            print "forming block", i, j, "of transfer matrix K"
            K_block_coo = QTheta_mtxs[i-j].tocoo()
#            print "    coo mtx:\n", K_block_coo
            for row_idx, col_idx, elem in zip(K_block_coo.row,K_block_coo.col,K_block_coo.data):
#                print "    appending elem..."
                K_list.append_csr_elem((i*ktn.n_nodes)+row_idx,(j*ktn.n_nodes)+col_idx, \
                    binomial_coeff(i,i-j)*elem)
    # final sum matrix
    final_vec = sorted([node.node_id-1 for node in ktn.A])
    for i in range(nmax+1):
        for j in final_vec:
            F_list.append_csr_elem(i,(i*ktn.n_nodes)+j,1.)
    K = K_list.return_csr_mtx()
    G = G_list.return_csr_mtx()
    F = F_list.return_csr_mtx()
    print "\nK:", K
    print "K nnz:", K.nnz
    return K, G, F, tau, eta, delS

''' get the jump matrix (transition probability matrix when the elements are branching probabilities) in
    CSR sparse format, and similar matrices for calculating moments of other dynamical quantities '''
def get_jump_mtxs(ktn,nmax):
    Q_list = List_CSR(ktn.n_nodes,ktn.n_nodes) # jump matrix (used to calculate moments for dynamical activity)
    Qs_mtxs = [List_CSR(ktn.n_nodes,ktn.n_nodes) for i in range(nmax+1)] # matrices used to calculate moments for path action
    for edge in ktn.edgelist:
        if edge.deadts: continue
        Q_list.append_csr_elem(edge.from_node.node_id-1,edge.to_node.node_id-1,edge.t)
        for j, Qs_list in enumerate(Qs_mtxs):
            Qs_list.append_csr_elem(edge.from_node.node_id-1,edge.to_node.node_id-1,-edge.t*(np.log(edge.t)**j))
    # ensure that rows of jump matrix sum to 1
    # ... quack
    Q = Q_list.return_csr_mtx()
    for i in range(nmax+1):
        Qs_mtxs[i] = Qs_mtxs[i].return_csr_mtx()
    return Q, Qs_mtxs

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
def manhart_morozov(ktn,nmax=2,eps=1.E-03):
    print "\n>>>>> performing the algorithm of Manhart & Morozov to calculate the moments for distributions of path statistics\n"
    if not (isinstance(ktn,Ktn) or isinstance(ktn,Coarse_ktn)): raise AttributeError
    if len(ktn.B)!= 1: raise AttributeError # only allow a single source node
    Analyse_ktn.calc_tbranch(ktn) # get the transition matrix as the matrix of branching probabilities (jump matrix)
    Q, Qs_mtxs = get_jump_mtxs(ktn,nmax)
    Theta_mtxs = get_holding_time_mtxs(ktn,nmax)
    K, G, F, tau, eta, delS = setup_transfer_mtxs(ktn,nmax,Q,Qs_mtxs,Theta_mtxs)

    print "\ninitial tau:\n", tau
    print "\njump matrix:\n", Q.toarray()
    for i in range(nmax+1):
        print "\nholding time Theta matrix #", i
        print Theta_mtxs[i].toarray()
    print "\ntransfer matrix K:\n", K.toarray()

    tau_lprev, eta_lprev = copy(tau), copy(eta)
    # main loop
    l=0
    t_L = np.zeros(shape=nmax+1,dtype=float) # accumulated path time probability distributions. Zeroth moment is path length
    path_times_f = open("path_times.dat","w")
    path_times_cum_f = open("path_times_cum.dat","w")
    path_times_f.write("# l / dynamical activity / 1st and higher moments of time distribution\n")
    path_times_cum_f.write("# l / cum path length prob / cum 1st and higher moments of time distribution\n")
    while True:
        print "iteration %i:" % (l+1)
        # calculate moment vectors of the current iteration
        tau_l = K.dot(tau_lprev)
        eta_l = G.dot(eta_lprev)
#        print type(tau_l), type(eta_l)
#        print "\ttau of current iteration:\n", tau_l
#        print "\t", [(i,tau_l[i]) for i in range(len(tau_l)) if tau_l[i]!=0.]
#        print "length of new transfer vectors:", tau_l.shape, eta_l.shape
        # update cumulative moment vectors 
        tau = tau+tau_l
        eta = eta+eta_l
        t_l = F.dot(tau_l)
        t_L = t_L+t_l
#        print "length of accumulated transfer vectors:", tau.shape, eta.shape
#        print t_l
        tau_lprev, eta_lprev = copy(tau_l), copy(eta_l)
        l+=1
        # write data to files
        path_times_f.write("%i" % l)
        path_times_cum_f.write("%i" % l)
        for i in range(nmax+1):
            path_times_f.write("  %1.12f" % t_l[i])
            path_times_cum_f.write("  %1.12f" % t_L[i])
        path_times_f.write("\n")
        path_times_cum_f.write("\n")
        print "\trho_L:", t_L[0]
        # check convergence
        if 1.-t_L[0]<eps and t_l[nmax]>0. and t_l[nmax]/t_L[nmax]<eps:
            break
#        if l>15: break
    path_times_f.close()
    path_times_cum_f.close()
