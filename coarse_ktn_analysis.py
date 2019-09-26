'''
Main Python script to estimate and analyse a coarse transition network
Usage:
python coarse_ktn_analysis.py <n_nodes> <n_edges> <n_comms>
requires 4x input data files

Daniel J. Sharpe
Sep 2019
'''

import numpy as np
from os.path import exists
from sys import argv
from scipy.sparse.linalg import eigs as eigs_iram
from scipy.sparse import csr_matrix

class Node(object):

    def __init__(self,node_id,ktn=None):
        self.node_id = node_id  # node ID. NB indexed from 1
        self.node_en = None     # node "energy"
        self.comm_id = None     # community ID. NB indexed from 0
        self.pi = None          # (log) node stationary probability
        self.mr = None          # reactive trajectory current
        self.qf = None          # node forward committor probability
        self.qb = None          # node backward committor probability
        self.k_esc = None       # escape rate from node (node out-degree)
        self.k_in = None        # sum of transition rates into node (node in-degree)
        self.t = None           # self-transition probability
        self.nbrlist = []       # list of neighbouring nodes via TO edges
        self.edgelist_in = []   # list of edges TO node
        self.edgelist_out = []  # list of edges FROM node
        self.ktn = ktn          # a reference to the Ktn object to which the Node belongs

    def __repr__(self):
        return self.__class__.__name__+"("+str(self.node_id)+")"

    def __str__(self):
        return "Node: %i  Community: %i  Stat prob: %.4f  Forward committor: %.4f" % ( \
               self.node_id, self.comm_id, self.pi, self.qf)

    def __lt__(self,other_node):
        if other_node.__class__.__name__ != "Node": raise AttributeError
        return self.node_en < other_node.node_en

    @property
    def node_attribs(self):
        return self.node_en,self.comm_id,self.pi

    @node_attribs.setter
    def node_attribs(self,vals):
        self.node_en, self.comm_id, self.pi = vals[0], vals[1], vals[2]

    @node_attribs.deleter
    def node_attribs(self):
        self.node_en, self.comm_id, self.pi = None, None, None
        self.nbrlist, self.edgelist_in, self.edgelist_out = [], [], []
        del self.tpt_vals

    @property
    def tpt_vals(self):
        return self.qf, self.qb, self.mr, self.t, self.k_esc, self.k_in

    ''' when updating the committor functions, calculate TPT quantities '''
    @tpt_vals.setter
    def tpt_vals(self,vals):
        if self.t is None: self.calc_node_t()
        if self.k_esc is None: self.calc_k_esc_in(mode=0)
        if self.k_in is None: self.calc_k_esc_in(mode=1)
        update_edges = False
        if self.qf != None: update_edges = True
        self.qf, self.qb = vals[0], vals[1]
        self.mr = self.calc_mr(self) # quack
        if update_edges:
            pass

    @tpt_vals.deleter
    def tpt_vals(self):
        print "deleting tpt_vals for node", self.node_id
        self.qf, self.qb, self.mr, self.t, self.k_esc = [None]*5

    @staticmethod
    def calc_mr(node,dbalance=False):
        if not Ktn.check_if_node(node): raise AttributeError
        if not dbalance:
            return node.pi*node.qf*node.qb
        else:
            return node.pi*node.qf*(1.-node.qf)

    ''' calculate self-transition probability for node '''
    def calc_node_t(self):
        sum_t_out = 0.
        for edge in self.edgelist_out:
            if edge.deadts: continue
            sum_t_out += edge.t
        self.t = 1.-sum_t_out

    ''' calculate escape rate or total in rate for node '''
    def calc_k_esc_in(self,mode):
        print "updating k_esc/k_in for node %i" % self.node_id
        edgelist, deg_obj = None, None
        if mode==0: edgelist, deg_obj = self.edgelist_out, self.k_esc
        elif mode==1: edgelist, deg_obj = self.edgelist_in, self.k_in
        sum_k = -float("inf")
        for edge in edgelist:
            if edge.deadts: continue
            sum_k = np.log(np.exp(sum_k)+np.exp(edge.k))
        deg_obj = sum_k

class Edge(object):

    def __init__(self,edge_id,ts_id,ktn=None):
        self.edge_id = edge_id   # edge ID (NB is unique) (NB indexed from 1)
        self.ts_id = ts_id       # transition state ID (NB edges are bidirectional, shared TS ID) (NB indexed from 1)
        self.ts_en = None        # transition state energy
        self.k = None            # (log) transition rate
        self.t = None            # transition probability
        self.j = None            # net flux
        self.f = None            # reactive flux
        self.fe = None           # net reactive flux
        self.deadts = False      # flag to indicate "dead" transition state
        self.to_node = None      # edge is: to_node <- from_node
        self.from_node = None
        self.__rev_edge = None   # reverse edge, corresponding to from_node<-to_node
        self.ktn = ktn           # a reference to the Ktn object to which the Edge belongs

    def __repr__(self):
        return self.__class__.__name__+"("+str(self.edge_id)+","+str(self.ts_id)+")"

    def __str__(self):
        return "Edge: %i  TS: %i  transition rate: %.4f  net flux: %.4f  net reactive flux: %.4f" % ( \
               self.edge_id, self.ts_id, self.k, self.j, self.fe)

    def __lt__(self,other_edge):
        if other_edge.__class__.__name__ != "Edge": raise AttributeError
        return self.ts_en < other_edge.ts_en

    ''' merge a pair of edges - only allow if start/end nodes are the same for both edges '''
    def __add__(self,other_edge):
        if self < other_edge: edge1, edge2 = self, other_edge
        else: edge1, edge2 = other_edge, self
        if not ((edge1.to_node==edge2.to_node) and (edge1.from_node==edge2.from_node)):
            raise AttributeError
        edge1.k += edge2.k
        if (edge1.t is not None) and (edge2.t is not None): edge1.t += edge2.t
        if (edge1.j is not None) and (edge2.j is not None): edge1.j += edge2.j
        if (edge1.f is not None) and (edge2.f is not None): edge1.f += edge2.f
        del edge2.edge_attribs

    @property
    def edge_attribs(self):
        return [self.ts_en,self.k,self.t,self.j,self.f,self.fe]

    @edge_attribs.setter
    def edge_attribs(self,vals):
        self.ts_en, self.k = vals[0], vals[1]

    @edge_attribs.deleter
    def edge_attribs(self):
        self.ts_en, self.k, self.t, self.j, self.f, self.fe = [None]*6
        self.deadts = True # flag the edge as having been the target of deletion
        try:
            del to_from_nodes
            del __rev_edge
        except UnboundLocalError:
            pass

    @property
    def to_from_nodes(self):
        if (self.to_node is None) or (self.from_node is None): raise AttributeError
        return [self.to_node.node_id,self.from_node.node_id]

    ''' set the TO and FROM nodes for this edge. args[1] is bool to indicate if nodes are not already connected '''
    @to_from_nodes.setter
    def to_from_nodes(self,args):
        for node in args[0]:
            if node.__class__.__name__ != "Node": raise AttributeError
        self.to_node, self.from_node = args[0][0], args[0][1]
        if (self.to_node.node_id==self.from_node.node_id): # self-loop: for KTNs indicates "dead" TS
            del self.edge_attribs
            return
        dup_edge, edge = None, self
        # check if edge is a duplicate of an existing edge (ie connects same pair of nodes)
        if (args[1] and (any(x.node_id==self.to_node.node_id for x in self.from_node.nbrlist) \
            or any(x.node_id==self.from_node.node_id for x in self.to_node.nbrlist))):
            dup_to_idx = next((i for i, x in enumerate(self.to_node.nbrlist) if x==self.from_node.node_id),None)
            dup_from_idx = next((i for i, x in enumerate(self.from_node.nbrlist) if x==self.to_node.node_id),None)
            if dup_to_idx is not None: dup_edge = self.ktn.edgelist[dup_to_idx]
            elif dup_from_idx is not None: dup_edge = self.ktn.edgelist[dup_from_idx]
            if dup_edge is None: raise AttributeError # this should not happen
            # delete the higher energy transition state
            if dup_edge < self:
                del self.edge_attribs
                return # this edge is now "dead", elems to nodes' nbrlist/edgelist_'s are never added
            else: # make the duplicate edge "dead" and remove entry in the relevant node's nbrlist
                del dup_edge.edge_attribs
                nbrlist, node = None, None
                if dup_to_idx is not None:
                    nbrlist, node = self.to_node.nbrlist, self.to_node
                else:
                    nbrlist, node = self.from_node.nbrlist, self.from_node
                nbrlist.pop(nbrlist.index(node.node_id)) # required to remove and set again so nbrlist/edgelist_ entries correspond
        if args[1] or dup_edge is not None: # update nbrlists of TO and FROM nodes
            self.to_node.nbrlist.append(self.from_node)
            self.from_node.nbrlist.append(self.to_node)
        edge.to_node.edgelist_in.append(self)
        edge.from_node.edgelist_out.append(self)

    @to_from_nodes.deleter
    def to_from_nodes(self):
        if not self.deadts: # here flag is indicating that the edge has been replaced with one
                            # connecting the same pair of nodes
            filter(lambda x: x.node_id==self.from_node.node_id,self.to_node.nbrlist)
            filter(lambda x: x.node_id==self.to_node.node_id,self.from_node.nbrlist)
        filter(lambda x: x.ts_id==self.ts_id,self.to_node.edgelist_in)
        filter(lambda x: x.ts_id==self.ts_id,self.from_node.edgelist_out)
        self.to_node, self.from_node = None, None

    @property
    def flow_vals(self):
        return [self.j,self.f,self.fe]

    @flow_vals.setter
    def flow_vals(self):
        if (self.to_node is None) or (self.from_node is None): raise AttributeError
        # quack
        dummy = Edge.calc_j
        # for these, need to return none if committors of nodes not set
        dummy = Edge.calc_f
        dummy = Edge.calc_fe

    @flow_vals.deleter
    def flow_vals(self):
        self.j, self.f, self.fe = None, None, None

    @staticmethod
    def calc_j(edge1,edge2):
        if not ((Ktn.check_if_edge(edge1)) or (Ktn.check_if_edge(edge2))): raise AttributeError
        if not Ktn.check_edge_reverse(edge1,edge2): raise AttributeError
        if ((edge1.k is None) or (edge2.k is None)): raise AttributeError
        j1 = (edge1.k*edge1.from_node.pi) - (edge2.k*edge2.from_node.pi)
        j2 = -j1
        return j1, j2

    @staticmethod
    def calc_f(node1,node2):
        if not ((Ktn.check_if_node(node1)) or (Ktn.check_if_node(node2))): raise AttributeError

    @staticmethod
    def calc_fe(edge1,edge2):
        if not ((Ktn.check_if_edge(edge1)) or (Ktn.check_if_edge(edge2))): raise AttributeError
        if not Ktn.check_edge_reverse(edge1,edge2): raise AttributeError
        if ((edge1.f is None) or (edge2.f is None)): raise AttributeError
        fe1 = max(0.,edge1.f-edge2.f)
        fe2 = max(0.,edge2.f-edge1.f)
        return fe1, fe2

    @property
    def rev_edge(self):
        return self.rev_edge

    @rev_edge.setter
    def rev_edge(self,edge):
        if not Ktn.check_if_edge(edge): raise AttributeError
        self.__rev_edge = edge

class Ktn(object):

    def __init__(self,n_nodes,n_edges,n_comms):
        self.n_nodes = n_nodes    # number of nodes
        self.n_edges = n_edges    # number of bidirectional edges
        self.n_comms = n_comms    # number of communities into which nodes are partitioned
        self.nodelist = [Node(i+1,ktn=self) for i in range(self.n_nodes)]
        self.edgelist = [Edge(i,((i-(i%2))/2)+1,ktn=self) for i in range(1,(2*self.n_edges)+1)]
        self.dbalance = False     # flag to indicate if detailed balance holds
        self.A = set()            # endpoint nodes in the set A (NB A<-B)
        self.B = set()            # endpoint nodes in the set B
        if not isinstance(self,Coarse_ktn): # do not do this for Coarse_ktn class
            self.comm_pi_vec = [-float("inf")]*self.n_comms # (log) stationary probabilities of communities

    def construct_ktn(self,comms,conns,pi,k,node_ens,ts_ens):
        if ((len(node_ens)!=self.n_nodes) or (len(ts_ens)!=self.n_edges) or (len(comms)!=self.n_nodes) \
            or (len(conns)!=self.n_edges) or (len(pi)!=self.n_nodes) \
            or (len(k)!=2*self.n_edges)): raise AttributeError
        for i in range(self.n_nodes):
            if comms[i] > self.n_comms-1: raise AttributeError
            self.nodelist[i].node_attribs = [node_ens[i],comms[i],pi[i]]
        for i in range(self.n_edges):
            self.edgelist[2*i].edge_attribs = [ts_ens[i],k[2*i]]
            self.edgelist[(2*i)+1].edge_attribs = [ts_ens[i],k[(2*i)+1]]
            # set edge connectivity
            self.edgelist[2*i].to_from_nodes = ([self.nodelist[conns[i][0]-1],self.nodelist[conns[i][1]-1]],True)
            self.edgelist[(2*i)+1].to_from_nodes = ([self.nodelist[conns[i][1]-1],self.nodelist[conns[i][0]-1]],False)
            self.edgelist[2*i].rev_edge = self.edgelist[(2*i)+1]
            self.edgelist[(2*i)+1].rev_edge = self.edgelist[2*i]
        self.renormalise_pi(mode=0) # check node stationary probabilities are normalised
        self.get_comm_stat_probs() # get stationary probabilities of communities
        print "calculating k_esc/k_in for full network:"
        for node in self.nodelist:
            node.calc_k_esc_in(0)
            node.calc_k_esc_in(1)

    ''' read (at least) node connectivity, stationary distribution and transition rates from files '''
    @staticmethod
    def read_ktn_info(n_nodes,n_edges):
        comms = Ktn.read_single_col("communities.dat",n_nodes)
        conns = Ktn.read_double_col("ts_conns.dat",n_edges)
        pi = Ktn.read_single_col("stat_prob.dat",n_nodes,fmt="float")
        k = Ktn.read_single_col("ts_weights.dat",2*n_edges,fmt="float")
        node_ens = Ktn.read_single_col("node_ens.dat",n_nodes,fmt="float") # optional
        ts_ens = Ktn.read_single_col("ts_ens.dat",n_edges,fmt="float") # optional
        return comms, conns, pi, k, node_ens, ts_ens

    ''' write the network to files in a format readable by Gephi '''
    def print_gephi_fmt(self,fmt=".gephi"):
        pass

    @staticmethod
    def read_single_col(fname,n_lines,fmt="int"):
        fmtfunc = int
        if fmt=="float": fmtfunc = float
        if not exists(fname): return [None]*n_lines
        with open(fname) as datafile:
            data = [fmtfunc(next(datafile)) for i in xrange(n_lines)]
        return data

    @staticmethod
    def read_double_col(fname,n_lines,fmt="int"):
        fmtfunc = int
        if fmt=="float": fmtfunc = float
        if not exists(fname): return [None]*n_lines
        data = [None]*n_lines
        with open(fname) as datafile:
            for i in xrange(n_lines):
                line = next(datafile).split()
                data[i] = [fmtfunc(line[0]),fmtfunc(line[1])]
        return data

    @staticmethod
    def check_if_node(node):
        return node.__class__.__name__=="Node"

    @staticmethod
    def check_if_edge(edge):
        return edge.__class__.__name__=="Edge"

    ''' check that edges are the "reverse" of one another, i<-j and j<-i '''
    @staticmethod
    def check_edge_reverse(edge1,edge2):
        if ((edge1.to_node is None) or (edge2.to_node is None)): raise AttributeError
        return ((edge1.to_node==edge2.from_node) and (edge2.to_node==edge1.from_node))

    def renormalise_mr(self):
        self.Zm = 0. # prob that a trajectory is reactive at a given instance in time
        for i in range(self.n_nodes):
            self.Zm += self.nodelist[i].mr
        for i in range(self.n_nodes):
            self.nodelist[i].mr *= 1./self.Zm

    def renormalise_pi(self,mode=1):
        tot_pi, tot_pi2 = -float("inf"), -float("inf") # accumulated stationary probability
        for i in range(self.n_nodes):
            tot_pi = np.log(np.exp(tot_pi)+np.exp(self.nodelist[i].pi))
        tot_pi = np.exp(tot_pi)
        if mode==0: # just check
            assert abs(tot_pi-1.) < 1.E-10
            return
        for i in range(self.n_nodes):
            self.nodelist[i].pi = np.log(np.exp(self.nodelist[i].pi)*(1./tot_pi))
            tot_pi2 = np.log(np.exp(tot_pi2)+np.exp(self.nodelist[i].pi))
        assert abs(np.exp(tot_pi2)-1.) < 1.E-10

    def get_comm_stat_probs(self):
        self.comm_pi_vec = [-float("inf")]*self.n_comms
        for node in self.nodelist:
            self.comm_pi_vec[node.comm_id] = np.log(np.exp(self.comm_pi_vec[node.comm_id]) + \
                np.exp(node.pi))
        print self.comm_pi_vec
        assert abs(sum([np.exp(comm_pi) for comm_pi in self.comm_pi_vec])-1.) < 1.E-10

    def construct_coarse_ktn(self):
        coarse_ktn = Coarse_ktn(self)
        return coarse_ktn

class Coarse_ktn(Ktn):

    def __init__(self,parent_ktn):
        if parent_ktn.__class__.__name__ != self.__class__.__bases__[0].__name__: raise AttributeError
        super(Coarse_ktn,self).__init__(parent_ktn.n_comms,parent_ktn.n_comms*(parent_ktn.n_comms-1),None)
        self.parent_ktn = parent_ktn
        self.construct_coarse_ktn

    ''' Construct a coarse network given the communities and inter-node transition rates for the full network
        from which it is to be derived.
        Note that a Coarse_ktn object has an empty edgelist that includes all inter-community transitions in a
        different order compared to Ktn objects read from a file.
        Note that Ktn.construct_coarse_ktn(...) is overridden here. Thus Ktn.construct_coarse_ktn() cannot be
        called from the Coarse_ktn derived class, as is desirable '''
    @property
    def construct_coarse_ktn(self):
        for i, node in enumerate(self.nodelist):
            node.pi = self.parent_ktn.comm_pi_vec[i]
        self.renormalise_pi(mode=0) # check node stationary probabilities are normalised
        for i, node1 in enumerate(self.nodelist):
            j=0
            for node2 in self.nodelist:
                if node1.node_id==node2.node_id: continue
#                print (i*self.n_nodes)+j+1, "FROM:", node1.node_id, "TO:", node2.node_id
                self.edgelist[(i*self.n_nodes)+j].to_from_nodes = ((node2,node1),node1.node_id<node2.node_id)
                j += 1
        for i in range(len(self.nodelist)):
            for j in range(i+1,len(self.nodelist)): # note i<j
                idx1 = (j*(self.n_nodes-1))+i   # edge index for j <- i
                idx2 = (i*(self.n_nodes-1))+j-1 # edge index for i <- j
                self.edgelist[idx1].rev_edge = self.edgelist[idx2]
                self.edgelist[idx2].rev_edge = self.edgelist[idx1]
        for node in self.parent_ktn.nodelist: # FROM node
            for edge in node.edgelist_out: # loop over all FROM edges
                if node.comm_id != edge.to_node.comm_id: # inter-community edge
                    idx = edge.to_node.comm_id*(self.n_nodes-1)+node.comm_id-\
                          (lambda x1, x2: (x1>x2 and 1 or 0))(node.comm_id,edge.to_node.comm_id)
                    coarse_edge = self.edgelist[idx]
                    if coarse_edge.k is None:
                        coarse_edge.k = np.exp(edge.k)
                    else:
                        coarse_edge.k = np.log(np.exp(coarse_edge.k)+np.exp(edge.k))
        for edge in self.edgelist:
            if edge.k is None: del edge.edge_attribs # mark inter-community edges with zero rate as "dead"
        for node in self.nodelist:
            node.calc_k_esc_in(0)
            node.calc_k_esc_in(1)

class Analyse_coarse_ktn(object):

    def __init__(self):
        pass

    ''' set up the transition rate matrix in CSR sparse format '''
    def setup_sp_k_mtx(self,ktn):
        K_row_idx, K_col_idx = [], [] # NB K[K_row_idx[i],K_col_idx[i]] = data[i]
        K_data = []
        nnz = 0 # count number of non-zero elements
        for i, node in enumerate(ktn.nodelist):
            diag_elem_idx = None # index of diagonal element in list
            sum_elems = 0. # sum of off-diagonal elems
            for j, nbr_node in enumerate(node.nbrlist):
                if nbr_node.node_id > i+1: # need to add an entry that will be the diag elem
                    diag_elem_idx = nnz
                    K_row_idx.append(i)
                    K_col_idx.append(i)
                    K_data.append(0.)
                    nnz += 1
                if node.edgelist_out[j].deadts: continue
                K_row_idx.append(i)
                K_col_idx.append(nbr_node.node_id-1)
                K_elem = np.exp(node.edgelist_out[j].k)
                K_data.append(K_elem)
                nnz += 1
                sum_elems += K_elem
            if diag_elem_idx is not None:
                K_data[diag_elem_idx] = -sum_elems
            else:
                K_row_idx.append(i)
                K_col_idx.append(i)
                K_data.append(-sum_elems)
                nnz += 1
        K_sp = csr_matrix((K_data,(K_row_idx,K_col_idx)),shape=(ktn.n_nodes,ktn.n_nodes),dtype=float)
        return K_sp

    ''' set up the transition rate matrix '''
    def setup_k_mtx(self,ktn):
        pass

    ''' calculate the transition matrix by matrix exponential '''
    def calc_t(self,tau):
        pass

    ''' calculate the linearised transition matrix '''
    def calc_tlin(self,tau):
        pass

    ''' calculate the k dominant eigenvalues and eigenvectors of a sparse matrix by the implicitly restarted Arnoldi method (IRAM) '''
    @staticmethod
    def calc_eig_iram(M,k,which_eigs="SM"):
        M_eigs, M_evecs = eigs_iram(M,k,which=which_eigs)
        M_evecs = np.transpose(M_evecs)
        M_evecs = np.array([M_evec for _,M_evec in sorted(zip(list(M_eigs),list(M_evecs)),key=lambda pair: pair[0])],dtype=float)
        M_eigs = np.array(sorted(list(M_eigs),reverse=True),dtype=float)
        return M_eigs, M_evecs

    ''' function to perform variational optimisation of the second dominant eigenvalue of
        the transition rate (and therefore of the transition) matrix, by perturbing the
        assigned communities and using a simulated annealing procedure '''
    def varopt_simann(self):
        pass

    @staticmethod
    def eigs_K_to_T(g,tau):
        return np.exp(g*tau)

if __name__=="__main__":

    ### TESTS ###
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

    ### MAIN ###
    n_nodes = int(argv[1])
    n_edges = int(argv[2])
    n_comms = int(argv[3])
    full_network = Ktn(n_nodes,n_edges,n_comms)
    comms, conns, pi, k, node_ens, ts_ens = Ktn.read_ktn_info(n_nodes,n_edges)
    full_network.construct_ktn(comms,conns,pi,k,node_ens,ts_ens)

    ### TEST KTN ###
    print "\n\n\n"
    print "nbrlist for node 333: ", full_network.nodelist[332].nbrlist
    print "edgelist_in for node 333: ", full_network.nodelist[332].edgelist_in
    print "edgelist_out for node 333: ", full_network.nodelist[332].edgelist_out
    print "stationary probabilities of communities:\n", [np.exp(x) for x in full_network.comm_pi_vec]
    print "\nk_esc for nodes:"
    for i in range(full_network.n_nodes): print full_network.nodelist[i].k_esc

    print "\ndominant eigenvalues of transition rate matrix:"
    analyser = Analyse_coarse_ktn()
    K_sp = analyser.setup_sp_k_mtx(full_network)
    K_sp_eigs, K_sp_evecs = Analyse_coarse_ktn.calc_eig_iram(K_sp,14)
    print K_sp_eigs

    print "\nforming the coarse matrix:"
    coarse_ktn = full_network.construct_coarse_ktn()
    print "no. of nodes:", coarse_ktn.n_nodes
    for i in range(coarse_ktn.n_nodes):
        print i+1, full_network.nodelist[i].k_esc, coarse_ktn.nodelist[i].k_esc
        print "\t", abs(coarse_ktn.nodelist[i].k_esc-full_network.nodelist[i].k_esc)>1.E-10
#    print "nodelist:", coarse_ktn.nodelist
#    print "edgelist:", coarse_ktn.edgelist
#    print "nbrlist for comm 2: ", coarse_ktn.nodelist[2].nbrlist # need to except dead TSs
#    print "edgelist_in for comm 2: ", coarse_ktn.nodelist[2].edgelist_in # ditto
#    print "edgelist_out for comm 2: ", coarse_ktn.nodelist[2].edgelist_out # ditto
    K_C_sp = analyser.setup_sp_k_mtx(coarse_ktn)
    K_C_sp_eigs, K_C_sp_evecs = Analyse_coarse_ktn.calc_eig_iram(K_C_sp,14)
    print K_C_sp_eigs
    
