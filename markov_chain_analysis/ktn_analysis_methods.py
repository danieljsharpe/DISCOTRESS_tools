'''
Python code containing methods for analysing kinetic transition networks

Daniel J. Sharpe
Sep 2019
'''

from __future__ import print_function
from ktn_structures import Node, Edge, Ktn, Coarse_ktn
import numpy as np
from copy import copy
from scipy.sparse.linalg import eigs as eigs_iram
from scipy.sparse import csr_matrix
from scipy.linalg import expm
from scipy import optimize # code is written for v0.17.0, also tested for v0.18.0
from os.path import exists

class Analyse_ktn(object):

    def __init__(self):
        pass

    ''' set up the transition rate matrix in CSR sparse format '''
    @staticmethod
    def setup_sp_k_mtx(ktn):
        if not isinstance(ktn,Ktn) and not isinstance(ktn,Coarse_ktn): raise RuntimeError
        K_row_idx, K_col_idx = [], [] # NB K[K_row_idx[i],K_col_idx[i]] = data[i]
        K_data = []
        # define lambda function to append an element
        append_csr_elem = lambda row_idx, col_idx, elem, row_vec=K_row_idx, col_vec=K_col_idx, data_vec=K_data: \
            (row_vec.append(row_idx), col_vec.append(col_idx), data_vec.append(elem))
        nnz = 0 # count number of non-zero elements
        for i, node in enumerate(ktn.nodelist):
            sum_elems = 0.
            diag_elem_idx = None # index of diagonal element in list
            if not node.edgelist_out:
                append_csr_elem(i,i,0.)
            for edge in node.edgelist_out:
                if diag_elem_idx is not None and edge.to_node.node_id-1 > i: # need to add an entry that is the diag elem
                    diag_elem_idx = nnz
                    append_csr_elem(i,i,-np.exp(node.k_esc))
                    nnz += 1
                    continue
                if edge.deadts: continue
                append_csr_elem(i,edge.to_node.node_id-1,np.exp(edge.k))
                nnz += 1
            if diag_elem_idx is None:
                append_csr_elem(i,i,-np.exp(node.k_esc))
                nnz += 1
        K_sp = csr_matrix((K_data,(K_row_idx,K_col_idx)),shape=(ktn.n_nodes,ktn.n_nodes),dtype=float)
        return K_sp

    ''' set up the transition rate matrix (not sparse) '''
    @staticmethod
    def setup_k_mtx(ktn):
        K = np.zeros((ktn.n_nodes,ktn.n_nodes),dtype=np.float128)
        for edge in ktn.edgelist:
            if edge.deadts: continue
            K[edge.from_node.node_id-1,edge.to_node.node_id-1] = np.exp(edge.k)
        for node in ktn.nodelist:
            K[node.node_id-1,node.node_id-1] = -np.exp(node.k_esc)
        for i in range(ktn.n_nodes):
            assert abs(np.sum(K[i,:])) < 1.E-05
        return K

    ''' set up the reverse transition rate matrix (not sparse) '''
    @staticmethod
    def setup_k_rev_mtx(ktn):
        K_rev = np.zeros((ktn.n_nodes,ktn.n_nodes),dtype=float)
        for edge in ktn.edgelist:
            if edge.deadts: continue
            K_rev[edge.to_node.node_id-1,edge.from_node.node_id-1] = np.exp(edge.from_node.pi-\
                edge.to_node.pi+edge.k)
        for i in range(ktn.n_nodes):
            K_rev[i,i] = -np.sum(K_rev[i,:])
        return K_rev

    ''' calculate the transition probability matrix by matrix exponential at a lag time tau. NB this transition
        probability matrix is less sparse than the corresponding transition rate matrix. Therefore this method
        can be used only when the Ktn data structure has edge entries for all (including unconnected) i-j pairs '''
    @staticmethod
    def calc_t(ktn,tau):
        pass

    ''' calculate the linearised transition probability matrix at a lag time tau. NB the linearised transition
        matrix has non-zero and zero entries at the same indices as the transition rate matrix, so the Ktn
        data structure can be updated directly. '''
    @staticmethod
    def calc_tlin(ktn,tau):
        for node in ktn.nodelist:
            assert tau <= 1./np.exp(node.k_esc) # must hold to obtain a proper stochastic matrix
            node.t = 1.-(tau*np.exp(node.k_esc))
        for edge in ktn.edgelist:
            if edge.deadts: continue
            edge.t = tau*np.exp(edge.k)
        ktn.tau = tau

    ''' calculate the transition probability matrix with the same eigenvectors as the rate matrix '''
    @staticmethod
    def calc_tsameeig(ktn,c,d):
        assert (0.<c and c<1.)
        cmax = float("-inf")
        for node in ktn.nodelist:
            if np.exp(node.k_esc) > cmax: cmax = np.exp(node.k_esc)
        print("calculating linearised transition probability matrix, effective lag time:",c/cmax)
        for node in ktn.nodelist: node.t = (-1.*(c/cmax)*np.exp(node.k_esc))+(1.*d)
        for edge in ktn.edgelist: edge.t = (c/cmax)*np.exp(edge.k)

    ''' calculate the transition probability matrix as the matrix of branching probabilities '''
    @staticmethod
    def calc_tbranch(ktn):
        for node in ktn.nodelist: node.t = 0.
        for edge in ktn.edgelist:
            if edge.deadts: continue
            edge.t = np.exp(edge.k-edge.from_node.k_esc)
        ktn.tau = 0. # indicates that the transition matrix has been calculated from the branching probabilities

    ''' dump the transition probabilities to a file "transnprobs.dat" '''
    @staticmethod
    def dump_tprobs(ktn):
        with open("transnprobs.dat","w") as tprobs_f:
            for edge in ktn.edgelist: tprobs_f.write("%1.25f\n" % edge.t)
            for node in ktn.nodelist: tprobs_f.write("%1.25f\n" % node.t)

    ''' calculate the k dominant eigenvalues and (normalised) eigenvectors of a sparse matrix by the
        implicitly restarted Arnoldi method (IRAM) '''
    @staticmethod
    def calc_eig_iram(M,k,which_eigs="SM"):
        M_eigs, M_evecs = eigs_iram(M,k,which=which_eigs)
        M_eigs, M_evecs = Analyse_ktn.sort_eigs(M_eigs,M_evecs)
        M_evecs = np.array([Analyse_ktn.normalise_vec(M_evec) for M_evec in M_evecs])
        return M_eigs, M_evecs

    @staticmethod
    def calc_eig_all(M,k=None):
        M_eigs, M_evecs = np.linalg.eig(M)
        M_eigs, M_evecs = Analyse_ktn.sort_eigs(M_eigs,M_evecs)
        if k is not None: M_eigs, M_evecs = M_eigs[:k+1], M_evecs[:k+1]
        M_evecs = np.array([Analyse_ktn.normalise_vec(M_evec) for M_evec in M_evecs])
        return M_eigs, M_evecs

    @staticmethod
    def sort_eigs(eigs,evecs):
        evecs = np.transpose(evecs)
        evecs = np.array([evec for _,evec in sorted(zip(list(eigs),list(evecs)),key=lambda pair: pair[0], \
            reverse=True)],dtype=float)
        eigs = np.array(sorted(list(eigs),reverse=True),dtype=float)
        return eigs, evecs

    @staticmethod
    def normalise_vec(vec):
        sumvec = np.sum([abs(x) for x in vec])
        vec *= 1./sumvec
        return vec

    ''' calculate committor function by chosen method and update Ktn data structure accordingly, including
        the calculation of TPT quantities for Node and Edge objects '''
    @staticmethod
    def calc_committors(ktn,method="linopt"):
        if not (isinstance(ktn,Ktn) or isinstance(ktn,Coarse_ktn)): raise AttributeError
        if method=="linopt":
            calc_committors_func = Analyse_ktn.calc_committors_linopt
        elif method=="sor":
            calc_committors_func = Analyse_ktn.calc_committors_sor
        else:
            raise RuntimeError
        qf = calc_committors_func(ktn,direction="f")
        if not ktn.dbalance:
            qb = calc_committors_func(ktn,direction="b")
        else:
            qb = np.array([1.-qf_i for qf_i in qf])
        ktn.update_all_tpt_vals(qf,qb)

    ''' calculate the committor functions by formulating a constrained linear optimisation problem '''
    @staticmethod
    def calc_committors_linopt(ktn,direction="f",seed=21):
        if not (isinstance(ktn,Ktn) or isinstance(ktn,Coarse_ktn)): raise AttributeError
        print("calculating committor functions by constrained linear optimisation...")
        np.random.seed(seed)
        if direction=="f": # calculate forward committor function (A<-B)
            start_set, final_set = ktn.B, ktn.A
            K = Analyse_ktn.setup_k_mtx(ktn)
        elif direction=="b": # calculate backward committor function (B<-A)
            start_set, final_set = ktn.A, ktn.B
            K = Analyse_ktn.setup_k_rev_mtx(ktn)
        else:
            raise RuntimeError
        q_constraints = [] # list of constraints. Each constraint is a dict.
        # nodes of starting/final sets have committor probabilities equal to zero/unity, respectively
        for node in start_set:
            constraint_func = (lambda i: (lambda x: x[i-1]))(node.node_id)
            q_constraints.append({"type": "eq", "fun": constraint_func})
        for node in final_set:
            constraint_func = (lambda i: (lambda x: x[i-1]-1.))(node.node_id)
            q_constraints.append({"type": "eq", "fun": constraint_func})
        # committor function satisfies 0 <= q_i <= 1 for all microstates i - can use these constraints in place
        #   of "bounds" kw argument to optimize.minimize()
#        q_constraints.append({"type": "ineq", "fun": lambda x: x[:]})
#        q_constraints.append({"type": "ineq", "fun": lambda x: -x+1.})
        # For all nodes not in A and B, the dot product of the committor vector with the corresponding row of
        #   the transition rate matrix must be equal to zero
        for node in ktn.nodelist:
            if node in ktn.A or node in ktn.B: continue
            constraint_func = (lambda i: (lambda x: np.dot(x,K[i-1,:])))(node.node_id)
            q_constraints.append({"type": "eq", "fun": constraint_func})
        x0 = np.random.rand(ktn.n_nodes)
        conopt_func = lambda x: np.dot(np.dot(K,x),np.dot(K,x)) # cast problem as linear eqn of form Ax=0
        q_res = optimize.minimize(conopt_func,x0=x0,method="SLSQP",bounds=[(0.,1.)]*ktn.n_nodes, \
            tol=1.E-8,constraints=q_constraints, \
            options={"maxiter": 500, "eps": 1.E-10, "ftol": 1.E-8, "iprint":3, "disp": True})
        if not q_res.success: raise RuntimeError
        q = q_res.x # vector of committor function values
        return q

    ''' calculate the committor functions by successive over-relaxation (SOR) '''
    @staticmethod
    def calc_committors_sor(ktn,direction="f"):
        if not (isinstance(ktn,Ktn) or isinstance(ktn,Coarse_ktn)): raise AttributeError
        pass

    ''' calculate the eigenvectors of the transition matrix and update this information in the data structure. If
        this function is called from a Coarse_ktn object, then both the coarse and parent KTN objects are updated '''
    def get_eigvecs(self):
        full_network, coarse_network = self, None
        n_nodes, n_comms = None, None # no. of nodes and of communities of full ktn, respectively
        if self.__class__.__name__=="Coarse_ktn":
            n_nodes, n_comms = self.parent_ktn.n_nodes, self.n_nodes
            coarse_network = self
            full_network = self.parent_ktn
        else:
            n_nodes, n_comms = self.n_nodes, self.n_comms
        K_sp_full = Analyse_ktn.setup_sp_k_mtx(full_network)
        eigvecs_full = Analyse_ktn.calc_eig_iram(K_sp_full,n_comms)[1]
        for i in range(n_nodes):
            node = full_network.nodelist[i]
            node.evec = [0.]*n_comms
            for j in range(n_comms):
                node.evec[j] = eigvecs_full[j,i]
        if coarse_network is None: return
        K_sp_coarse = Analyse_ktn.setup_k_mtx(coarse_network)
        eigvecs_coarse = Analyse_ktn.calc_eig_all(K_sp_coarse,n_comms)[1]
        for i in range(n_comms):
            node = coarse_network.nodelist[i]
            node.evec = [0.]*n_comms
            for j in range(n_comms):
                node.evec[j] = eigvecs_coarse[j,i]
        coarse_network.get_evec_errors(full_network.n_comms)

    ''' function to perform variational optimisation of the second dominant eigenvalue of
        the transition rate (and therefore of the transition) matrix, by perturbing the
        assigned communities and using a simulated annealing procedure '''
    def varopt_simann(self,coarse_ktn,nsteps,seed=21):
        # print inter-community transition rates
        for node in coarse_ktn.nodelist:
            if node.node_id==41: quit()
            print("comm: ", node.node_id-1)
            for edge in node.edgelist_out:
                if edge.k is None: continue
                print("    TO comm: ", edge.to_node.node_id-1, " ln k: ", edge.k)
        quit()

        np.random.seed(seed)
        K_C = self.setup_k_mtx(coarse_ktn)
        K_C = K_C.astype(float)
        K_C_copy = None
        K_C_eigs, K_C_evecs = Analyse_ktn.calc_eig_all(K_C)
        lambda2_prev = K_C_eigs[1]
        niter=0
        while (niter<nsteps):
            K_C_copy = copy(K_C)
            nodes_to_update = [0]*coarse_ktn.n_nodes # mask of coarse_ktn nodes whose escape times need updating
            # find an inter-community edge
            found_edge, edge = False, None
            while not found_edge:
                edge_id = np.random.randint(1,2*coarse_ktn.parent_ktn.n_edges+1)
                edge = coarse_ktn.parent_ktn.edgelist[edge_id-1]
                if edge.deadts: continue
                if edge.from_node.comm_id!=edge.to_node.comm_id: found_edge = True
            nodes_to_update[edge.from_node.comm_id] = 1
            nodes_to_update[edge.to_node.comm_id] = 1
            # swap the community of a single boundary node
            pick_node = np.random.random_integers(0,1)
            if pick_node==1: edge = edge.rev_edge
            old_from_comm = edge.from_node.comm_id
            edge.from_node.comm_id = edge.to_node.comm_id
            old_from_comm_pi_prev = coarse_ktn.nodelist[old_from_comm].pi
            old_to_comm_pi_prev = coarse_ktn.nodelist[edge.from_node.comm_id].pi
            for fn_edge in edge.from_node.edgelist_out: # scan neighbours FROM edge.from_node
                if fn_edge.deadts: continue
                nodes_to_update[fn_edge.to_node.comm_id] = 1
                # transition is a new inter-community edge / inter-community edge now connects this different pair
                if fn_edge.to_node.comm_id!=edge.from_node.comm_id:
                    K_C[edge.from_node.comm_id,fn_edge.to_node.comm_id] += np.exp(edge.from_node.pi-\
                        coarse_ktn.nodelist[edge.from_node.comm_id].pi+fn_edge.k)
                    K_C[fn_edge.to_node.comm_id,edge.from_node.comm_id] += np.exp(fn_edge.to_node.pi-\
                        coarse_ktn.nodelist[fn_edge.to_node.comm_id].pi+fn_edge.rev_edge.k)
                # transition is no longer an inter-community edge / inter-community edge now connects a different pair
                if fn_edge.to_node.comm_id!=old_from_comm:
                    K_C[old_from_comm,fn_edge.to_node.comm_id] -= \
                        np.exp(edge.from_node.pi-coarse_ktn.nodelist[old_from_comm].pi+fn_edge.k)
                    K_C[fn_edge.to_node.comm_id,old_from_comm] -= \
                        np.exp(fn_edge.to_node.pi-coarse_ktn.nodelist[fn_edge.to_node.comm_id].pi+fn_edge.rev_edge.k)
            # update the stationary probabilities of nodes [communities] of the Coarse_ktn object
            coarse_ktn.nodelist[old_from_comm].pi = np.log(np.exp(coarse_ktn.nodelist[old_from_comm].pi)-\
                np.exp(edge.from_node.pi))
            coarse_ktn.nodelist[edge.from_node.comm_id].pi = np.log(np.exp(\
                coarse_ktn.nodelist[edge.from_node.comm_id].pi)+np.exp(edge.from_node.pi))
            # account for the population changes of the two perturbed communities in the inter-community rates
            K_C[old_from_comm,:] *= np.exp(old_from_comm_pi_prev-coarse_ktn.nodelist[old_from_comm].pi)
            K_C[edge.from_node.comm_id,:] *= np.exp(old_to_comm_pi_prev-coarse_ktn.nodelist[edge.from_node.comm_id].pi)
            # update the escape times of communities
            gen = (node for i, node in enumerate(coarse_ktn.nodelist) if nodes_to_update[i])
            for node in gen:
                K_C[node.node_id-1,node.node_id-1] = 0.
                K_C[node.node_id-1,node.node_id-1] = -np.sum(K_C[node.node_id-1,:])
            K_C_eigs, K_C_evecs = Analyse_ktn.calc_eig_all(K_C)
            if lambda2_prev < K_C_eigs[1]: # second dominant eigenvalue of rate matrix has increased, accept move
                print("accepting step %i: %f -> %f" % (niter+1,lambda2_prev,K_C_eigs[1]))
                lambda2_prev = K_C_eigs[1]
                # update the escape rates and transition rates in the Coarse_ktn data structure
                # NB also need to add/subtract the new edges to the inter-community transition rate in the data structure!
                for out_edge in coarse_ktn.nodelist[old_from_comm].edgelist_out:
                    if out_edge.deadts: continue
                    out_edge.k = np.log(np.exp(out_edge.k)*np.exp(\
                                 old_from_comm_pi_prev-coarse_ktn.nodelist[old_from_comm].pi))
                for out_edge in coarse_ktn.nodelist[edge.from_node.comm_id].edgelist_out:
                    if out_edge.deadts: continue
                    out_edge.k = np.log(np.exp(out_edge.k)*np.exp(\
                                 old_to_comm_pi_prev-coarse_ktn.nodelist[edge.from_node.comm_id].pi))
                coarse_ktn.nodelist[old_from_comm].calc_k_esc_in(mode=1)
                coarse_ktn.nodelist[edge.from_node.comm_id].calc_k_esc_in(mode=1)
                coarse_ktn.parent_ktn.comm_sz_vec[old_from_comm] -= 1
                coarse_ktn.parent_ktn.comm_sz_vec[edge.from_node.comm_id] += 1
            else: # reject move
                K_C = copy(K_C_copy)
                coarse_ktn.nodelist[old_from_comm].pi = old_from_comm_pi_prev
                coarse_ktn.nodelist[edge.from_node.comm_id].pi = old_to_comm_pi_prev
                edge.from_node.comm_id = old_from_comm
            niter += 1
        # update the stationary probability vector of communities in the Ktn object
        # quack
        print("finished variational optimisation of communities, writing new communities to file...")
        with open("communities_new.dat","w") as newcomms_f:
            for node in coarse_ktn.nodelist:
                print("ID: ", node.node_id)
                newcomms_f.write("%i\n" % node.comm_id)
        return K_C

    ''' perform an analysis of the series of isocommittor cuts on a KTN. The isocommittor cuts are defined at intervals of
        1/(ncuts+1). ncuts=1 calculates the TSE only '''
    @staticmethod
    def isocommittor_cut_analysis(ktn,ncuts=1,dircn="f",writedata=True):
        if not isinstance(ktn,Ktn) and not isistance(ktn,Coarse_ktn): raise RuntimeError
        if not ktn.tpt_calc_done: raise AttributeError # need committor functions and other TPT values for this analysis
        incmt=1./float(ncuts+1)
        alpha_vals = [0.+(incmt*float(i)) for i in range(1,ncuts+1)]
        cut_edges_all = [[] for i in range(ncuts)]
        for i, alpha_val in enumerate(alpha_vals):
            cut_edges = Analyse_ktn.get_isocommittor_cut(ktn,alpha_val,dircn=dircn)[1]
            cut_edges_all[i] = cut_edges
            if not writedata: continue
            if exists("cut_flux.alpha"+str(alpha_val)+".dat"): raise RuntimeError
            with open("cut_flux.alpha"+str(alpha_val)+".dat","w") as cut_flux_f:
                cut_flux_f.write("# cumulative flux      cumulative relative flux      alpha=%f\n" % alpha_val)
                for j, cut_edge in enumerate(cut_edges_all[i]):
                    cut_flux_f.write("%i    %1.12f    %1.12f\n" % (j,cut_edge[1],cut_edge[2]))

    ''' calculate the total A<-B reactive flux and retrieve the set of edges that form the isocommittor cut defined by
        a committor function value equal to alpha. alpha=0.5 defines the transition state ensemble (TSE) '''
    @staticmethod
    def get_isocommittor_cut(ktn,alpha,cumvals=True,dircn="f"):
        if not isinstance(ktn,Ktn) and not isistance(ktn,Coarse_ktn): raise RuntimeError
        if alpha<=0. or alpha>=1.: raise RuntimeError
        if not ktn.tpt_calc_done: raise AttributeError # need committor functions and other TPT values for this analysis
        J = 0. # reactive A-B flux
        cut_edges = [] # list of edges that constitute the isocommittor cut in format (edge,J_ij,rel J_ij)
        for edge in ktn.edgelist:
            if edge.deadts: continue
            edge_in_cut = None
            if dircn=="f" and edge.to_node.qf>alpha and edge.from_node.qf<alpha: # forwards (A<-B)
                edge_in_cut = edge
                comm_diff = edge_in_cut.to_node.qf-edge_in_cut.from_node.qf # difference in committors across edge of cut
            elif dircn=="b" and edge.from_node.qb>alpha and edge.to_node.qb<alpha: # backwards (B<-A)
                edge_in_cut = edge.rev_edge
                comm_diff = edge_in_cut.to_node.qb-edge_in_cut.from_node.qb
            if edge_in_cut is None: continue
            J_ij = np.exp(edge_in_cut.k+edge_in_cut.from_node.pi)*comm_diff
            cut_edges.append([edge_in_cut,J_ij,J_ij])
            J += J_ij
        for cut_edge in cut_edges: cut_edge[2] *= 1./J
        cut_edges = sorted(cut_edges,key=lambda x: x[2],reverse=True)
        if cumvals: # calculate cumulative values for reactive fluxes along edges
            for i in range(1,len(cut_edges)):
                cut_edges[i][1] += cut_edges[i-1][1]
                cut_edges[i][2] += cut_edges[i-1][2]
        return J, cut_edges

    ''' write the edge costs required for a shortest paths algorithm based on the reactive flux along individual edges,
        in the same format as the edge_weights.dat input file, to the file flux_weights.dat.
        The actual edge costs to be used (eg in DISCOTRESS) are -\ln[val], where val is the value written to the file.
        In this sense, val is a global quantity analogous to the transition probs T_{ij} when using "local" edge costs
        -\ln T_{ij} in a shortest paths algorithm.
        Note that edges are unidirectional in the reactive flux representation; zero values correspond to inf edge cost '''
    @staticmethod
    def write_flux_edgecosts(ktn):
        nodeinB = [False]*ktn.n_nodes
        norm_factors = [0.]*ktn.n_nodes # for edges from a node not in B, the edge costs incl a factor due to the tot reactive flux assocd with the from node
        for node in ktn.B:
            nodeinB[node.node_id-1] = True;
            norm_factors[node.node_id-1]=1.
        for node in ktn.A: # flux along edges from nodes in A is always zero, choose arbitrary nonzero number to avoid divide by zero
            norm_factors[node.node_id-1]=1.
        for edge in ktn.edgelist:
            if nodeinB[edge.from_node.node_id-1]: continue
            norm_factors[edge.from_node.node_id-1] += edge.fe
        foo = open("flux_weights.dat","w")
        for i in range(ktn.n_edges):
            c1 = ktn.edgelist[2*i].fe/norm_factors[ktn.edgelist[2*i].from_node.node_id-1] # edge cost for forward edge
            c2 = ktn.edgelist[(2*i)+1].fe/norm_factors[ktn.edgelist[(2*i)+1].from_node.node_id-1] # edge cost for reverse edge
            foo.write("%s      %s\n" % ("{:.32e}".format(c1),"{:.32e}".format(c2)))
        foo.close()

    ''' calculate the Kullback-Liebler divergence between two transition networks. This relative entropy measure
        quantifies the difference in transition probability distributions between two transition probability matrices.
        NB the KL divergence is asymmetric '''
    @staticmethod
    def calc_kl_div(ktn1,ktn2):
        if (not isinstance(ktn1,Ktn) and not isinstance(ktn1,Coarse_ktn)) or \
           (not isinstance(ktn2,Ktn) and not isinstance(ktn2,Coarse_ktn)): raise RuntimeError
        if ktn1.tau is None or ktn2.tau is None: # this analysis requires that transition probabilities have been calculated
            raise AttributeError
        # quack

    ''' calculate the Jensen-Shannon divergence between two transition networks. The JS divergence is essentially a
        symmetrised equivalent of the KL divergence '''
    @staticmethod
    def calc_js_div(ktn1,ktn2,wts=[0.5,0.5]):
        if not sum(wts)==1.: raise RuntimeError
        if (not isinstance(ktn1,Ktn) and not isinstance(ktn1,Coarse_ktn)) or \
           (not isinstance(ktn2,Ktn) and not isinstance(ktn2,Coarse_ktn)): raise RuntimeError
        if ktn1.tau is None or ktn2.tau is None: raise AttributeError

    ''' calculate the approximate JS divergence between two related transition networks by computing a weighted sum over
        the surprisal values for each node. A node has high surprisal when the associated transition probabilities differ
        greatly between the two TNs. Note that the two Ktn objects must have the same number of nodes '''
    @staticmethod
    def calc_surprisal(ktn1,ktn2,writedata=True):
        if (not isinstance(ktn1,Ktn) and not isinstance(ktn1,Coarse_ktn)) or \
           (not isinstance(ktn2,Ktn) and not isinstance(ktn2,Coarse_ktn)): raise RuntimeError
        if ktn1.tau is None or ktn2.tau is None: raise AttributeError
        if ktn1.n_nodes != ktn2.n_nodes: raise AttributeError
        ktn_comb = ktn1+ktn2
        for i in range(ktn1.n_nodes):
            print(i, ktn_comb.nodelist[i].t, ktn1.nodelist[i].t, ktn2.nodelist[i].t)
            assert ktn_comb.nodelist[i].t==(ktn1.nodelist[i].t+ktn2.nodelist[i].t)/2.
            assert ktn_comb.nodelist[i].pi==np.log((np.exp(ktn1.nodelist[i].pi)+np.exp(ktn2.nodelist[i].pi))/2.)
            for j in range(len(ktn1.nodelist[i].edgelist_out)):
                print(">>>>>", ktn_comb.nodelist[i].edgelist_out[j].t, ktn1.nodelist[i].edgelist_out[j].t, ktn2.nodelist[i].edgelist_out[j].t)
                assert ktn_comb.nodelist[i].edgelist_out[j].t==(ktn1.nodelist[i].edgelist_out[j].t+ktn2.nodelist[i].edgelist_out[j].t)/2.
        H_i_vals_1 = [Analyse_ktn.calc_entropy_rate_node(node) for node in ktn1.nodelist]
        H_i_vals_2 = [Analyse_ktn.calc_entropy_rate_node(node) for node in ktn2.nodelist]
        H_i_vals_comb = [Analyse_ktn.calc_entropy_rate_node(node) for node in ktn_comb.nodelist]
        print("populations:")
        print([np.exp(ktn1.nodelist[i].pi) for i in range(ktn1.n_nodes)])
        print([np.exp(ktn2.nodelist[i].pi) for i in range(ktn2.n_nodes)])
        print("MSM entropy rates:")
        print(H_i_vals_1)
        print(H_i_vals_2)
        print(H_i_vals_comb)
        # calculate surprisal values
        print(H_i_vals_comb[2])
        print((np.exp(ktn1.nodelist[2].pi-np.log(np.exp(ktn1.nodelist[2].pi)+np.exp(ktn2.nodelist[2].pi)))*H_i_vals_1[2]))
        print((np.exp(ktn2.nodelist[2].pi-np.log(np.exp(ktn1.nodelist[2].pi)+np.exp(ktn2.nodelist[2].pi)))*H_i_vals_1[2]))
        s_i_vals = [H_i_vals_comb[i]-\
                    (np.exp(ktn1.nodelist[i].pi-np.log(np.exp(ktn1.nodelist[i].pi)+np.exp(ktn2.nodelist[i].pi)))*H_i_vals_1[i])-\
                    (np.exp(ktn2.nodelist[i].pi-np.log(np.exp(ktn1.nodelist[i].pi)+np.exp(ktn2.nodelist[i].pi)))*H_i_vals_2[i])\
                    for i in range(ktn1.n_nodes)]
        print("surprisal values:")
        print(s_i_vals)
        pi_comb_vals = [(np.exp(ktn1.nodelist[i].pi)+np.exp(ktn2.nodelist[i].pi))/2. for i in range(ktn1.n_nodes)]
        print("avg. populations:")
        print(pi_comb_vals)
        djs_approx_vals = [pi_comb_vals[i]*s_i_vals[i] for i in range(ktn1.n_nodes)]
        # update Node's of ktn1 object with surprisal values
        for i, node in enumerate(ktn1.nodelist): node.s = s_i_vals[i]
        if not writedata: return
        with open("surprisal.dat","w") as surprisal_f:
            surprisal_f.write("# average stat prob      surprisal      approx. JS divergence\n")
            surprisal_f.write("# total JS divergence: %1.12f\n" % sum(djs_approx_vals))
            for i in range(ktn1.n_nodes):
                surprisal_f.write("%1.12f    %1.12f    %1.12f\n" % (pi_comb_vals[i],s_i_vals[i],djs_approx_vals[i]))

    ''' calculate the entropy rate for a node in a transition network '''
    @staticmethod
    def calc_entropy_rate_node(node):
        if not isinstance(node,Node): raise RuntimeError
        H = node.t*np.log(node.t) # entropy rate
        for edge in node.edgelist_out:
            if edge.deadts: continue
            H += edge.t*np.log(edge.t)
        return -H

    @staticmethod
    def eigs_K_to_T(g,tau):
        return np.exp(g*tau)

