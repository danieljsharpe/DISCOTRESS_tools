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

class Node(object):

    def __init__(self,node_id):
        self.node_id = node_id  # node ID
        self.node_en = None     # node "energy"
        self.comm_id = None     # community ID
        self.deg_in = None      # node in-degree
        self.deg_out = None     # node out-degree
        self.pi = None          # node stationary probability
        self.mr = None          # reactive trajectory current
        self.qf = None          # node forward committor probability
        self.qb = None          # node backward committor probability
        self.nbrlist = []       # list of neighbouring nodes
        self.edgelist_in = []   # list of edges TO node
        self.edgelist_out = []  # list of edges FROM node

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
        return self.qf, self.qb, self.mr

    ''' when updating the committor functions, calculate TPT quantities '''
    @tpt_vals.setter
    def tpt_vals(self,vals):
        update_edges = False
        if self.qf != None: update_edges = True
        self.qf, self.qb = vals[0], vals[1]
        self.mr = self.calc_mr(self) # quack
        if update_edges:
            pass

    @tpt_vals.deleter
    def tpt_vals(self):
        self.qf, self.qb, self.mr = None, None, None

    @staticmethod
    def calc_mr(node,dbalance=False):
        if not Ktn.check_if_node(node): raise AttributeError
        if not dbalance:
            return node.pi*node.qf*node.qb
        else:
            return node.pi*node.qf*(1.-node.qf)

class Edge(object):

    def __init__(self,edge_id,ts_id):
        self.edge_id = edge_id   # edge ID (NB is unique)
        self.ts_id = ts_id       # transition state ID (NB edges are bidirectional, shared TS ID)
        self.ts_en = None        # transition state energy
        self.k = None            # transition rate
        self.t = None            # transition probability
        self.j = None            # net flux
        self.f = None            # reactive flux
        self.fe = None           # net reactive flux
        self.deadts = False      # flag to indicate "dead" transition state
        self.to_node = None
        self.from_node = None
        self.__rev_edge = None   # reverse edge, corresponding to from_node<-to_node

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
        del to_from_nodes
        del __rev_edge

    @property
    def to_from_nodes(self):
        if (self.to_node is None) or (self.from_node is None): raise AttributeError
        return [self.to_node.node_id,self.from_node.node_id]

    ''' set the TO and FROM nodes for this edge '''
    @to_from_nodes.setter
    def to_from_nodes(self,nodes):
        for node in nodes:
            if node.__class__.__name__ != "Node": raise AttributeError
        self.to_node, self.from_node = nodes[0], nodes[1]
        self.to_node.nbrlist.append(self.from_node)
        self.from_node.nbrlist.append(self.to_node)
        self.to_node.edgelist_in.append(self)
        self.from_node.edgelist_out.append(self)

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
        self.nodelist = [Node(i) for i in range(self.n_nodes)]
        self.edgelist = [Edge(i,i) for i in range(2*self.n_edges)]
        self.dbalance = False     # flag to indicate if detailed balance holds
        self.A = set()            # endpoint nodes in the set A (NB A<-B)
        self.B = set()            # endpoint nodes in the set B

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
            self.edgelist[2*i].to_from_nodes = [self.nodelist[conns[i][0]-1],self.nodelist[conns[i][1]-1]]
            self.edgelist[(2*i)+1].to_from_nodes = [self.nodelist[conns[i][1]-1],self.nodelist[conns[i][0]-1]]

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
        self.Zm = 0 # prob that a trajectory is reactive at a given instance in time
        for i in range(self.n_nodes):
            self.Zm += self.nodelist[i].mr
        for i in range(self.n_nodes):
            self.nodelist[i].mr *= 1./self.Zm

class Coarse_ktn(Ktn):

    pass

class Analyse_coarse_ktn(object):

    def __init__(self):
        pass

    ''' set up the transition rate matrix '''
    def setup_k_mtx(self,ktn):
        pass

    ''' set up the sparse transition rate matrix '''
    def setup_sp_k_mtx(self,ktn):
        pass

    ''' calculate the transition matrix by matrix exponential '''
    def calc_t(self,tau):
        pass

    ''' function to perform variational optimisation of the second dominant eigenvalue of
        the transition rate (and therefore of the transition) matrix, by perturbing the
        assigned communities and using a simulated annealing procedure '''
    def varopt_simann(self):
        pass

if __name__=="__main__":

    ### TESTS ###
    mynode1 = Node(1)
    mynode1.node_attribs = [-0.2,1,0.45]
    mynode1.tpt_vals = [0.3,0.7]
    mynode2 = Node(6)
    mynode2.node_attribs = [-0.4,2,0.30]
    myedge1 = Edge(5,5)
    myedge1.to_from_nodes = [mynode1,mynode2]
    mynode1.node_id = 2
    print "edge #1 to/from:", myedge1.to_from_nodes
    print "ID of first IN edge of node 1:", mynode1.edgelist_in[0].edge_id
    print repr(mynode1), "\n", str(mynode1)
    del mynode1.node_attribs
    print "forward committor for node 1 has now been deleted. qf:", mynode1.qf

    mynode3 = Node(3)
    mynode3.node_attribs = [-0.5,4,0.25]
    myedge2 = Edge(8,8)
    myedge2.to_from_nodes = [mynode1,mynode3]
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
