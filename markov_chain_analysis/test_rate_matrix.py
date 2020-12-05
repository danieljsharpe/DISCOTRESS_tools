'''
Python script to construct a toy transition rate matrix for a stochastic block model network,
calculate the corresponding transition matrix, and calculate its dominant eigenvectors.
Then estimate a coarse-grained transition rate matrix given the known communities, and do the
same for this matrix.
The communities can be chosen deliberately wrong (i.e. not the same as those used to construct the
stochastic block model), and variationally optimising (maximising) the second eigenvalue can find
the `optimal' partition (that which gives the most appropriate description of the second slowest process
on the network - for a 'well-behaved' network this should rediscover the communities and optimise all
eigenvalues of the coarse transition matrix.)
'''

from __future__ import print_function
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from copy import deepcopy

# eigenvalues of transition matrix are associated with characteristic timescales
def get_timescales(evals,tau_lag):
    char_times = np.zeros((np.shape(evals)[0]),dtype=float)
    # note that we ignore the zero eigenvalue, associated with infinite time (stationary distribution)
    for i, eigval in enumerate(evals[1:]):
        char_times[i+1] = -tau_lag/np.log(eigval)
    return char_times

n_V = 500 # no. of vertices
n_C = 5 # no. of communities
# probability of edges existing between vertices of communities i and j. Should be a symmetric matrix
# (diagonal elements are larger probs - nodes in same community likely to be connected)
# this `stochastic block model' reduces to the Erdos-Renyi random graph model if p_ij = p for all i,j
p_arr = np.array([[0.60, 0.05, 0.05, 0.00, 0.00],
                  [0.05, 0.50, 0.05, 0.05, 0.00],
                  [0.05, 0.05, 0.40, 0.05, 0.05],
                  [0.00, 0.05, 0.05, 0.55, 0.05],
                  [0.00, 0.00, 0.05, 0.05, 0.50]])
seed = 19 # random seed
mean_k = 10. # mean rate (elements of rate matrix are Gaussian-distributed)
sigma_k = 2. # std dev of rate matrix elements
tau = 0.002 # lag time for estimation of transition matrix
tau_C = 0.002 # lag time for estimatation of coarse transition matrix
k = 10 # no. of eigenvectors of the (full) transition matrix to calculate
n_it_var = 1200 # no. of steps in variational optimisation of second eigenvalue
dump_network = False # write the graph to files and quit Y/N
# Erdos-Renyi random graph model:
# as p*n_V -> c > 1, then the graph will almost surely have a single large connected
# component of order n_V (percolation transition threshold exceeded)

print("\nlag time: ", tau, "\tno. of vertices", n_V, "\tno. of communities", n_C)

np.random.seed(seed)
# equal no. of vertices in each community
c_idx = []
for c_id in range(n_C): c_idx.extend([c_id]*(n_V/n_C))
# set up adjacency matrix according to stochastic block model and assign rates
K = np.zeros((n_V,n_V))
for i in range(n_V):
    for j in range(i+1,n_V):
        p = p_arr[c_idx[i],c_idx[j]]
        if np.random.rand() < p: # edge exists
            # NB transition rate matrix is not, in general, symmetric
            for nodes in [[i,j],[j,i]]:
                while True:
                    K[nodes[0],nodes[1]] = np.random.normal(mean_k,sigma_k)
                    if K[nodes[0],nodes[1]] > 0.: break

# now set diagonal elements of transition rate matrix
for i in range(n_V):
    K[i,i] = -np.sum(K[i,:])

# print("Transition rate matrix K:\n", K)

# transition matrix is matrix exponential of transition rate matrix and is estimated at a lag time tau
T = expm(tau*K)
# NB if do: T = expm(tau*np.transpose(K)) then T is a column-stochastic matrix, i.e. columns sum to one,
# eigenvalues are the same as the corresponding row-stochastic matrix.

# rows of transition matrix must sum to unity
# print("Transition matrix T:\n", T)
for i in range(n_V):
    assert abs(np.sum(T[i,:])-1.) < 1.0E-10, "Error, row %i of transition matrix does not sum to 1" % i

# remember np.dot(K,pi) = 0, i.e. stationary distribution is the (normalised) right eigenvector of the
# transition rate matrix that corresponds to eigenvalue = 0
# the transition rate matrix has a unique zero eigenvalue. The corresponding (normalised) eigenvector is the
# stationary probability (pi). All other eigenvalues are < 0.
K_evals, K_evecs = eigs(K,k,which="SM")
pi = np.array(K_evecs[:,0]*(1./np.sum(K_evecs[:,0])),dtype=float)
# print("stationary distribution pi:\n", pi)
assert abs(np.sum(pi)-1.) < 1.0E-10, "Error, node stationary probabilities do not sum to 1"

if dump_network:
    sp_f = open("stat_prob_sbm.dat","w")
    comms_f = open("communities_sbm.dat","w")
    ts_conns_f = open("ts_conns_sbm.dat","w")
    ts_weights_f = open("ts_weights_sbm.dat","w")
    for i in range(n_V):
        sp_f.write("%5.24f\n" % np.log(pi[i]))
        comms_f.write("%i\n" % c_idx[i])
        for j in range(i+1,n_V):
            if K[i,j]==0.: continue
            ts_conns_f.write("%i %i\n" % (i+1,j+1))
            ts_weights_f.write("%5.24f\n%5.24f\n" % (np.log(K[i,j]),np.log(K[j,i])))
    sp_f.close()
    comms_f.close()
    ts_conns_f.close()
    ts_weights_f.close()
    quit()

balance = True
for i in range(n_V):
    for j in range(i+1,n_V):
        if abs((K[i,j]*pi[i])-K[j,i]*pi[j]) < 1.0E-10:
            print("Detailed balance is NOT satisfied")
            balance = False
            break
    if not balance: break
if balance:
    print("Detailed balance IS satisfied")

# calculate k eigenvectors and eigenvalues of transition matrix by the Lanczos algorithm (a power method)
T_evals, T_evecs = eigs(T,k)
T_evals = np.array(T_evals,dtype=float)
T_evecs = np.array(T_evecs,dtype=float)
# calculate characteristic timescales
char_t_full = get_timescales(T_evals,tau)
# write eigenvectors and eigenvalues to file
# note that there are n_C dominant eigenvalues (which includes the first eigenvalue eval_1 = 1.) that correspond
# to each of the inter-community transitions. The corresponding eigenvectors give information on what is
# happening in these slow processes
with open("evals.dat","w") as eval_f:
    for T_eval in T_evals:
        eval_f.write("%1.14f\n" % T_eval)
with open("evals_K.dat","w") as eval_K_f:
    for K_eval in K_evals:
        eval_K_f.write("%1.14f\n" % K_eval)
with open("char_times.dat","w") as ct_f:
    for t in char_t_full:
        ct_f.write("%1.14f\n" % t)
for i in range(k):
    T_evec = T_evecs[:,i]
    with open("evec."+str(i)+".dat","w") as evec_f:
        for node_id, evec_val in enumerate(T_evec):
            evec_f.write("%5i   %1.10f\n" % (node_id, evec_val))
# NB by the Perron-Frobenius theorem, the largest eigenvalue of the transition matrix is equal to unity
# all other eigenvalues are < 1

# assign the c_idx (array of communities to which nodes belong) incorrectly
# (compared to the c_idx used to build the stochastic block model) - i.e. construction of coarse-grained
# transition rate matrix is not optimal
#'''
c_idx = []
c_idx.extend([0]*140)
c_idx.extend([1]*90)
c_idx.extend([2]*50)
c_idx.extend([3]*160)
c_idx.extend([4]*60)
#'''

# construct a coarse-grained transition rate matrix, estimated from rate constants for inter-community
# transitions in the full transition rate matrix
pi_C = np.zeros(n_C,dtype=float) # probability vector for communities
K_C = np.zeros((n_C,n_C),dtype=float) # transition rate matrix for communities
for i in range(n_V):
    pi_C[c_idx[i]] += pi[i]
for i in range(n_V):
    for j in range(n_V):
        if c_idx[i]==c_idx[j]: continue
        K_C[c_idx[i],c_idx[j]] += (pi[i]/pi_C[c_idx[i]])*K[i,j]
for i in range(n_C):
    K_C[i,i] = -np.sum(K_C[i,:])
print("\ninitial coarse-grained transition rate matrix K_C:\n", K_C)
K_C_evals, K_C_evecs = np.linalg.eig(K_C)

# calculate and process coarse-grained transition matrix
T_C = expm(tau_C*K_C)
print("initial coarse-grained transition matrix T_C:\n", T_C)
for i in range(n_C):
    assert abs(np.sum(T_C[i,:])-1.) < 1.0E-08, "Error, row %i of coarse transition matrix does not sum to 1" % i
print("initial stationary distribution vector pi:\n", pi_C)
# NB coarse matrix is small enough to calculate all eigenvectors, so not using power method here
T_C_evals, T_C_evecs = np.linalg.eig(T_C)
T_C_evecs = np.transpose(T_C_evecs)
T_C_evecs = np.array([T_C_evec for _,T_C_evec in sorted(zip(list(T_C_evals),list(T_C_evecs)),key=lambda pair: pair[0])])
T_C_evals = np.array(sorted(list(T_C_evals),reverse=True),dtype=float)
char_t_C = get_timescales(T_C_evals,tau_C)
# write files for the estimated (initial) transition and transition rate matrices
with open("evals.est.dat","w") as eval_f:
    for T_C_eval in T_C_evals:
        eval_f.write("%1.14f\n" % T_C_eval)
with open("char_times.est.dat","w") as ct_f:
    for t in char_t_C:
        ct_f.write("%1.14f\n" % t)
with open("evals_K.est.dat","w") as eval_K_f:
    for K_C_eval in K_C_evals:
        eval_K_f.write("%1.14f\n" % K_C_eval)

# variationally optimise the second eigenvalue by perturbing the definition of the sets (communities)
n_change = 1 # no. of edges to change at a time
K_C_var = deepcopy(K_C)
T_C_var = deepcopy(T_C)
pi_C_var = deepcopy(pi_C)
c_idx_var = deepcopy(c_idx)
K_C_var_evals = deepcopy(K_C_evals)
T_C_var_evals = deepcopy(T_C_evals)
char_t_C_var = deepcopy(char_t_C)
n_it = 0
n_success = 0
T_C_eval2_curr = T_C_evals[1]
eval_K_prog_f = open("eval_K_prog.dat","w+")
eval_T_prog_f = open("eval_T_prog.dat","w+")
print("\n\nBeginning variational optimisation of second dominant eigenvalue of transition matrix...\n")
while n_it < n_it_var:
    K_C_var_old = deepcopy(K_C_var)
    T_C_var_old = deepcopy(T_C_var)
    pi_C_var_old = deepcopy(pi_C_var)
    K_C_var_evals_old = deepcopy(K_C_var_evals)
    T_C_var_evals_old = deepcopy(T_C_var_evals)
    char_t_C_var_old = deepcopy(char_t_C_var)
    c_idx_var_old = deepcopy(c_idx_var)
    # choose pairs of edges at random that connect two communities
    edge_changes = []
    for i in range(n_change):
        found_edge = False # flag for if found inter-community edge
        while not found_edge:
            node_1 = np.random.randint(0,n_V)
            node_2 = np.random.randint(0,n_V)
            if node_1 == node_2: continue
            if K[node_1,node_2] != 0. and c_idx_var[node_1] != c_idx_var[node_2]: found_edge = True
        edge_changes.append(np.array([node_1,node_2]))
    while edge_changes:
        # choose one of the two nodes to assign to the `opposite' community
        node_pair = edge_changes.pop()
        pick_node = np.random.randint(0,2)
        if pick_node == 0: unpick_node = 1
        elif pick_node == 1: unpick_node = 0
        old_c_idx = c_idx_var[node_pair[pick_node]] # old community index for the node that has swapped communities
        c_idx_var[node_pair[pick_node]] = c_idx_var[node_pair[unpick_node]]
        # update the coarse-grained transition rate matrix
        disjoint = True # the node is disjoint if it is not connected to another node of the community it is supposed to be a member of
        for i in range(n_V): # update contributions of inter-community rate constants
            if K[i,node_pair[pick_node]] > 0.: # found an edge that was or is an inter-community edge
                # note: for pi_C values we use the old values (before perturbing the cluster) and correct later
                if c_idx_var[i] == old_c_idx: # this transition is now an inter-community transition
                    K_C_var[c_idx_var[i],c_idx_var[node_pair[pick_node]]] += K[i,node_pair[pick_node]]*(pi[i]/pi_C_var[c_idx_var[i]])
                    K_C_var[c_idx_var[node_pair[pick_node]],c_idx_var[i]] += K[node_pair[pick_node],i]*(pi[node_pair[pick_node]]/pi_C_var[c_idx_var[node_pair[pick_node]]])
                elif c_idx_var[i] == c_idx_var[node_pair[pick_node]]: # this transition is no longer an inter-community transition
                    K_C_var[c_idx_var[i],old_c_idx] -= K[i,node_pair[pick_node]]*(pi[i]/pi_C_var[c_idx_var[i]])
                    K_C_var[old_c_idx,c_idx_var[i]] -= K[node_pair[pick_node],i]*(pi[node_pair[pick_node]]/pi_C_var[old_c_idx])
                    if disjoint: disjoint = False
                else: # transition was an inter-community edge to another community
                    K_C_var[c_idx_var[i],old_c_idx] -= K[i,node_pair[pick_node]]*(pi[i]/pi_C_var[c_idx_var[i]])
                    K_C_var[old_c_idx,c_idx_var[i]] -= K[node_pair[pick_node],i]*(pi[node_pair[pick_node]]/pi_C_var[old_c_idx])
                    K_C_var[c_idx_var[i],c_idx_var[node_pair[pick_node]]] += K[i,node_pair[pick_node]]*(pi[i]/pi_C_var[c_idx_var[i]])
                    K_C_var[c_idx_var[node_pair[pick_node]],c_idx_var[i]] += K[node_pair[pick_node],i]*(pi[node_pair[pick_node]]/pi_C_var[c_idx_var[node_pair[pick_node]]])
        if disjoint: quit("Node %i is disjoint from its community %i" % (node_pair[pick_node],c_idx_var[node_pair[pick_node]]))
        # account for population change of the two perturbed communities
        pi_C_var[old_c_idx] -= pi[node_pair[pick_node]]
        pi_C_var[c_idx_var[node_pair[pick_node]]] += pi[node_pair[pick_node]]
        K_C_var[old_c_idx,:] *= pi_C_var_old[old_c_idx]/pi_C_var[old_c_idx]
        K_C_var[c_idx_var[node_pair[pick_node]],:] *= pi_C_var_old[c_idx_var[node_pair[pick_node]]]/pi_C_var[c_idx_var[node_pair[pick_node]]]
    # update diagonal elements of coarse-grained transition rate matrix
    for i in range(n_C):
        K_C_var[i,i] = 0.
        K_C_var[i,i] = -np.sum(K_C_var[i,:])
    # calculate updated coarse-grained transition matrix
    T_C_var = expm(tau_C*K_C_var)
    for i in range(n_C):
        assert abs(np.sum(T_C_var[i,:])-1.) < 1.0E-08, "row %i of T_C_var does not sum to one" % i
    T_C_var_evals, T_C_var_evecs = np.linalg.eig(T_C_var)
    T_C_var_evals = sorted(T_C_var_evals,reverse=True)
    K_C_var_evals, K_C_var_evecs = np.linalg.eig(K_C_var)
    K_C_var_evals = sorted(K_C_var_evals,reverse=True)
    T_C_var_evals = np.array(sorted(list(T_C_var_evals),reverse=True),dtype=float)
    char_t_C_var = get_timescales(T_C_var_evals,tau_C)
    char_t_amt_impv = np.sum(np.array([char_t_C_var[i] - char_t_C_var_old[i] for i in range(1,n_C)]))
    evals_amt_impv = np.sum(np.array([T_C_var_evals[i] - T_C_var_evals_old[i] for i in range(1,n_C)]))
    # acceptance criterion - comment out the criteria you don't want to use
    if T_C_var_evals[1] < T_C_eval2_curr: # second eigenvalue has decreased, reject change
#    if evals_amt_impv < 0.: # net decrease in the eigenvalues of the coarse matrix, reject change
#    if char_t_amt_impv < 0.: # net decrease in the characteristic timescales of the coarse matrix, reject change
        K_C_var = deepcopy(K_C_var_old)
        T_C_var = deepcopy(T_C_var_old)
        pi_C_var = deepcopy(pi_C_var_old)
        K_C_var_evals = deepcopy(K_C_var_evals_old)
        T_C_var_evals = deepcopy(T_C_var_evals_old)
        char_t_C_var = deepcopy(char_t_C_var_old)
        c_idx_var = deepcopy(c_idx_var_old)
    else: # second eigenvalue of transition matrix has increased, accept change
        print(n_it, "\taccepting change:", T_C_var_evals[1], T_C_eval2_curr)
        T_C_eval2_curr = T_C_var_evals[1]
        n_success += 1
    # append the eigenvalues of the coarse transition and rate matrices to data files recording progress
    eval_T_prog_f.write("%5i" % n_it)
    for T_C_eval in T_C_var_evals:
        eval_T_prog_f.write("   %1.6f" % T_C_eval)
    eval_T_prog_f.write("\n")
    eval_K_prog_f.write("%5i" % n_it)
    for K_C_eval in K_C_var_evals:
        eval_K_prog_f.write("   %1.6f" % K_C_eval)
    eval_K_prog_f.write("\n")
    n_it += 1

eval_T_prog_f.close()
eval_K_prog_f.close()
print("\nNumber of successful community updates:", n_success)
print("Number of nodes in each community:")
n_comm = [0]*n_C
for i in range(n_V): n_comm[c_idx_var[i]] += 1
print(n_comm)

char_t_C_var = get_timescales(T_C_var_evals,tau_C)
with open("char_times.var.dat","w") as ct_f:
    for t in char_t_C_var:
        ct_f.write("%1.14f\n" % t)
with open("evals.var.dat","w") as eval_f:
    for T_C_eval in T_C_var_evals:
        eval_f.write("%1.14f\n" % T_C_eval)
with open("evals_K.var.dat","w") as eval_K_f:
    for K_C_eval in K_C_var_evals:
        eval_K_f.write("%1.14f\n" % K_C_eval)

print("K_C is now:\n", K_C_var)
print("\nT_C is now:\n", T_C_var)
print("\npi_C is now:\n", pi_C_var)
