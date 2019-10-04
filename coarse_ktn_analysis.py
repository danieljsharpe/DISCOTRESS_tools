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
from copy import copy
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
        self.evec = [0.]        # list of values for the dominant eigenvectors
        if ktn is not None and ktn.__class__.__name__=="Ktn":
            self.evec_err = [0.]    # list of errors associated with eigenvectors in parent vs coarse KTN
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
        if (self.k_esc is None) or (self.k_in is None): self.calc_k_esc_in()
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

    ''' calculate escape rate and total in rate for node '''
    def calc_k_esc_in(self,mode=0):
        sum_k_esc, sum_k_in = -float("inf"), -float("inf")
        for edge in self.edgelist_out:
            if edge.deadts: continue
            sum_k_esc = np.log(np.exp(sum_k_esc)+np.exp(edge.k))
        if mode==1: return # only calculate escape rate
        for edge in self.edgelist_in:
            if edge.deadts: continue
            sum_k_in = np.log(np.exp(sum_k_in)+np.exp(edge.k))
        self.k_esc = sum_k_esc
        self.k_in = sum_k_in

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
        return self.__rev_edge

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
            self.edgelist[2*i].to_from_nodes = ([self.nodelist[conns[i][1]-1],self.nodelist[conns[i][0]-1]],True)
            self.edgelist[(2*i)+1].to_from_nodes = ([self.nodelist[conns[i][0]-1],self.nodelist[conns[i][1]-1]],False)
            self.edgelist[2*i].rev_edge = self.edgelist[(2*i)+1]
            self.edgelist[(2*i)+1].rev_edge = self.edgelist[2*i]
        self.renormalise_pi(mode=0) # check node stationary probabilities are normalised
        self.get_comm_stat_probs() # get stationary probabilities of communities
        for node in self.nodelist:
            node.calc_k_esc_in()

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
    def print_gephi_fmt(self,fmt="csv",mode=0,evec_idx=0):
        if exists("ktn_nodes."+fmt) or exists("ktn_edges."+fmt): raise RuntimeError
        ktn_nodes_f = open("ktn_nodes."+fmt,"w")
        ktn_edges_f = open("ktn_edges."+fmt,"w")
        if fmt=="csv": ktn_nodes_f.write("Id,Label,Energy,Community,pi,evec,mr,qf,qb")
        if fmt=="csv": ktn_edges_f.write("Source,Target,Weight,Type,Energy,k,t,j,f,fe")
        if fmt=="csv":
            for node in self.nodelist:
                ktn_nodes_f.write(str(node.node_id)+","+str(node.node_id)+","+str(node.node_en)+","+str(node.comm_id)+","+\
                    str(node.pi)+","+str(node.mr)+","+str(node.qf)+","+str(node.qb)+","+str(node.evec[evec_idx]))

                if self.__class__.__name__ == "Coarse_ktn":
                    ktn_nodes_f.write(","+str(node.evec_err[evec_idx]))
                ktn_nodes_f.write("\n")
        ktn_nodes_f.close()
        if fmt=="csv" and mode==0: # directed edges (direction determined by net flux)
            for edge in self.edgelist:
                if edge.deadts: continue
                if mode==0 and edge.fe==0.: continue # directed edges, direction determined by net reactive flux
                if mode==1 and edge.j<0.: continue # directed edges, direction determined by net flux
                ktn_edges_f.write(str(edge.from_node.node_id)+","+str(edge.to_node.node_id)+","+str(edge.k)+","+str(edge.ts_en)+\
                    "directed,"+str(k)+","+str(t)+","+str(j)+","+str(f)+","+str(fe)+"\n")
        ktn_edges_f.close()

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
        assert abs(sum([np.exp(comm_pi) for comm_pi in self.comm_pi_vec])-1.) < 1.E-10

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
        K_sp_full = Analyse_coarse_ktn.setup_sp_k_mtx(full_network)
        eigvecs_full = Analyse_coarse_ktn.calc_eig_iram(K_sp_full,n_comms)[1]
        for i in range(n_nodes):
            node = full_network.nodelist[i]
            node.evec = [0.]*n_comms
            for j in range(n_comms):
                node.evec[j] = eigvecs_full[j,i]
        if coarse_network is None: return
        K_sp_coarse = Analyse_coarse_ktn.setup_k_mtx(coarse_network)
        eigvecs_coarse = Analyse_coarse_ktn.calc_eig_all(K_sp_coarse,n_comms)[1]
        for i in range(n_comms):
            node = coarse_network.nodelist[i]
            node.evec = [0.]*n_comms
            for j in range(n_comms):
                node.evec[j] = eigvecs_coarse[j,i]
        coarse_network.get_evec_errors(full_network.n_comms)

    def construct_coarse_ktn(self):
        coarse_ktn = Coarse_ktn(self)
        return coarse_ktn

    ''' dump relevant node information to files: eigenvectors, eigenvector errors, communities. If called
        from a Coarse_ktn object, then data for both the coarse and parent KTN objects is written '''
    def write_nodes_info(self):
        full_network, coarse_network = self, None
        n_comms = None # no. of communities of full ktn
        if self.__class__.__name__ == "Coarse_ktn":
            n_comms = self.n_nodes
            coarse_network = self
            full_network = self.parent_ktn
        else:
            n_comms = self.n_comms
        for i in range(n_comms):
            if exists("evec."+str(i+1)+".full.dat") or exists("evec."+str(i+1)+".coarse.dat") \
               or exists("evec."+str(i+1)+".err.dat"): raise RuntimeError
            evec_full_f = open("evec."+str(i+1)+".full.dat","w")
            if coarse_network is not None:
                evec_coarse_f = open("evec."+str(i+1)+".coarse.dat","w")
                evec_err_f = open("evec."+str(i+1)+".err.dat","w")
            for node in full_network.nodelist:
                evec_full_f.write("%1.12f\n" % node.evec[i])
                if coarse_network is not None:
                    evec_coarse_f.write("%1.12f\n" % coarse_network.nodelist[node.comm_id].evec[i])
                    evec_err_f.write("%1.12f\n" % node.evec_err[i])
            evec_full_f.close()
            if coarse_network is not None:
                evec_coarse_f.close()
                evec_err_f.close()
        if exists("communities_new.dat"): raise RuntimeError
        with open("communities_new.dat","w") as comms_new_f:
            for node in full_network.nodelist:
                comms_new_f.write(str(node.comm_id)+"\n")

class Coarse_ktn(Ktn):

    def __init__(self,parent_ktn):
        if parent_ktn.__class__.__name__ != self.__class__.__bases__[0].__name__: raise AttributeError
        super(Coarse_ktn,self).__init__(parent_ktn.n_comms,parent_ktn.n_comms*(parent_ktn.n_comms-1)/2,None)
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
        for i, node1 in enumerate(self.nodelist): # initialise TO and FROM nodes of j<-i edges
            j=0
            for node2 in self.nodelist:
                if node1.node_id!=node2.node_id:
                    idx = (i*(self.n_nodes-1))+j-(lambda x1, x2: (x1<x2 and 1 or 0))(i,j)
                    self.edgelist[idx].to_from_nodes = ((node2,node1),node1.node_id<node2.node_id)
                j += 1
        for i in range(len(self.nodelist)):
            for j in range(i+1,len(self.nodelist)): # note i<j
                idx1 = (i*(self.n_nodes-1))+j-1 # edge index for j <- i
                idx2 = (j*(self.n_nodes-1))+i   # edge index for i <- j
                self.edgelist[idx1].rev_edge = self.edgelist[idx2]
                self.edgelist[idx2].rev_edge = self.edgelist[idx1]
        for node in self.parent_ktn.nodelist: # FROM node
            for edge in node.edgelist_out: # loop over all FROM edges
                if node.comm_id != edge.to_node.comm_id: # inter-community edge
                    idx = (node.comm_id*(self.n_nodes-1))+edge.to_node.comm_id-\
                          (lambda x1, x2: (x1<x2 and 1 or 0))(node.comm_id,edge.to_node.comm_id)
                    coarse_edge = self.edgelist[idx]
                    if coarse_edge.k is None:
                        coarse_edge.k = node.pi-self.nodelist[node.comm_id].pi+edge.k
                    else:
                        coarse_edge.k = np.log(np.exp(coarse_edge.k)+np.exp(node.pi-self.nodelist[node.comm_id].pi+edge.k))
        for edge in self.edgelist:
            if edge.k is None:
                del edge.edge_attribs # mark inter-community edges with zero rate as "dead"
        for node in self.nodelist:
            node.calc_k_esc_in()

    ''' calculate errors in the n dominant eigenvectors of the transition matrix for the parent KTN compared
        to the coarsened ktn '''
    def get_evec_errors(self,n):
        for node in self.parent_ktn.nodelist:
            node.evec_err = [0.]*n
            for j in range(n):
                node.evec_err[j] = abs(node.evec[j]-self.nodelist[node.comm_id].evec[j])

class Analyse_coarse_ktn(object):

    def __init__(self):
        pass

    ''' set up the transition rate matrix in CSR sparse format '''
    @staticmethod
    def setup_sp_k_mtx(ktn):
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
        K = np.zeros((ktn.n_nodes,ktn.n_nodes),dtype=float)
        for edge in ktn.edgelist:
            if edge.deadts: continue
            K[edge.from_node.node_id-1,edge.to_node.node_id-1] = np.exp(edge.k)
        for node in ktn.nodelist:
            K[node.node_id-1,node.node_id-1] = -np.exp(node.k_esc)
        for i in range(ktn.n_nodes):
            assert abs(np.sum(K[i,:])) < 1.E-10
        return K
        
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
        M_eigs, M_evecs = Analyse_coarse_ktn.sort_eigs(M_eigs,M_evecs)
        return M_eigs, M_evecs

    @staticmethod
    def calc_eig_all(M,k=None):
        M_eigs, M_evecs = np.linalg.eig(M)
        M_eigs, M_evecs = Analyse_coarse_ktn.sort_eigs(M_eigs,M_evecs)
        if k is not None: M_eigs, M_evecs = M_eigs[:k+1], M_evecs[:k+1]
        return M_eigs, M_evecs

    @staticmethod
    def sort_eigs(eigs,evecs):
        evecs = np.transpose(evecs)
        evecs = np.array([evec for _,evec in sorted(zip(list(eigs),list(evecs)),key=lambda pair: pair[0], \
            reverse=True)],dtype=float)
        eigs = np.array(sorted(list(eigs),reverse=True),dtype=float)
        return eigs, evecs

    ''' function to perform variational optimisation of the second dominant eigenvalue of
        the transition rate (and therefore of the transition) matrix, by perturbing the
        assigned communities and using a simulated annealing procedure '''
    def varopt_simann(self,coarse_ktn,nsteps,seed=21):
        np.random.seed(seed)
        K_C = self.setup_k_mtx(coarse_ktn)
        K_C_copy = None
        K_C_eigs, K_C_evecs = Analyse_coarse_ktn.calc_eig_all(K_C)
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
            K_C_eigs, K_C_evecs = Analyse_coarse_ktn.calc_eig_all(K_C)
            if lambda2_prev < K_C_eigs[1]: # second dominant eigenvalue of rate matrix has increased, accept move
                print "accepting step %i: %f -> %f" % (niter+1,lambda2_prev,K_C_eigs[1])
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
            else: # reject move
                K_C = copy(K_C_copy)
                coarse_ktn.nodelist[old_from_comm].pi = old_from_comm_pi_prev
                coarse_ktn.nodelist[edge.from_node.comm_id].pi = old_to_comm_pi_prev
                edge.from_node.comm_id = old_from_comm
            niter += 1
        # update the stationary probability vector of communities in the Ktn object

        return K_C

    def isocommittor_surfaces(self):
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
    '''
    print "\n\n\n"
    print "nbrlist for node 333: ", full_network.nodelist[332].nbrlist
    print "edgelist_in for node 333: ", full_network.nodelist[332].edgelist_in
    print "edgelist_out for node 333: ", full_network.nodelist[332].edgelist_out
    print "stationary probabilities of communities:\n", [np.exp(x) for x in full_network.comm_pi_vec]
    print "out edges for node 1:", len(full_network.nodelist[0].edgelist_out)
    '''

    print "\ndominant eigenvalues of transition rate matrix:"
    analyser = Analyse_coarse_ktn()
    K_sp = Analyse_coarse_ktn.setup_sp_k_mtx(full_network)
    K_sp_eigs, K_sp_evecs = Analyse_coarse_ktn.calc_eig_iram(K_sp,7)
    print K_sp_eigs
    
    tau = 1.E+1
    print "\n eigenvalues of transition matrix, lag time:", tau
    print [Analyse_coarse_ktn.eigs_K_to_T(g,tau) for g in K_sp_eigs]

    print "\n characteristic timescales of full matrix:"
    print [1./eig for eig in K_sp_eigs]

    print "\nforming the coarse matrix:"
    coarse_ktn = full_network.construct_coarse_ktn()
    '''
    print "no. of nodes:", coarse_ktn.n_nodes
    print "nodelist:", coarse_ktn.nodelist
    print "edgelist:", coarse_ktn.edgelist
    print "nbrlist for comm 2: ", coarse_ktn.nodelist[2].nbrlist # need to except dead TSs
    print "edgelist_in for comm 2: ", coarse_ktn.nodelist[2].edgelist_in # ditto
    print "edgelist_out for comm 2: ", coarse_ktn.nodelist[2].edgelist_out # ditto
    '''
    print "stationary probabilities of coarse nodes:\n", [np.exp(x.pi) for x in coarse_ktn.nodelist]

    print "\neigenvalues of coarse matrix (sparse):"
    K_C_sp = Analyse_coarse_ktn.setup_sp_k_mtx(coarse_ktn)
    K_C_sp_eigs, K_C_sp_evecs = Analyse_coarse_ktn.calc_eig_iram(K_C_sp,3)
    print K_C_sp_eigs
    print "\neigenvalues of coarse matrix (not sparse):"
    K_C = Analyse_coarse_ktn.setup_k_mtx(coarse_ktn)
    K_C_eigs, K_C_evecs = Analyse_coarse_ktn.calc_eig_all(K_C)
    print K_C_eigs

    '''
    print "\n doing variational optimisation of coarse rate matrix:"
    K_C_opt = analyser.varopt_simann(coarse_ktn,5000)
    print "\neigenvalues of coarse matrix (after var opt procedure):"
    K_C_opt_eigs, K_C_opt_evecs = Analyse_coarse_ktn.calc_eig_all(K_C_opt)
    print K_C_opt_eigs
    print "\n characteristic timescales of coarse matrix:"
    print [1./eig for eig in K_C_opt_eigs]
    print "stationary probabilities of coarse nodes:\n", [np.exp(x.pi) for x in coarse_ktn.nodelist]
    '''

    coarse_ktn.get_eigvecs()
    coarse_ktn.write_nodes_info()
