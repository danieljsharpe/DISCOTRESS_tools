'''
Main Python script to estimate and analyse a coarse transition network

Daniel J. Sharpe
Sep 2019
'''

import numpy as np

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
        self.mr = self.calc_mr(self.qf,self.qb) # quack
        if update_edges:
            pass

    @tpt_vals.deleter
    def tpt_vals(self):
        self.qf, self.qb, self.mr = None, None, None

    @staticmethod
    def calc_mr(qf,qb):
        pass

class Edge(object):

    def __init__(self,edge_id,ts_id):
        self.edge_id = edge_id   # edge ID (NB is unique)
        self.ts_id = ts_id       # transition state ID (NB edges are bidirectional)
        self.ts_en = None        # transition state energy
        self.k = None            # transition rate
        self.t = None            # transition probability
        self.j = None            # net flux
        self.f = None            # reactive flux
        self.fe = None           # net reactive flux
        self.deadts = False      # flag to indicate "dead" transition state
        self.to_node = None
        self.from_node = None

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
        self.ts_en, self.k, self.t = vals[0], vals[1], vals[2]

    @edge_attribs.deleter
    def edge_attribs(self):
        self.ts_en, self.k, self.t, self.j, self.f, self.fe = [None]*6
        self.deadts = True # flag the edge as having been the target of deletion
        del to_from_nodes

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
        pass

    @staticmethod
    def calc_j(node1,node2):
        if not ((Edge.check_if_node(node1)) or (Edge.check_if_node(node2))): raise AttributeError

    @staticmethod
    def calc_f(node1,node2):
        if not ((Edge.check_if_node(node1)) or (Edge.check_if_node(node2))): raise AttributeError

    @staticmethod
    def calc_fe(node1,node2):
        if not ((Edge.check_if_node(node1)) or (Edge.check_if_node(node2))): raise AttributeError

    @staticmethod
    def check_if_node(node):
        return node.__class__.__name__=="Node"

class Ktn(object):

    def __init__(self,n_nodes,n_edges):
        self.n_nodes = n_nodes    # number of nodes
        self.nodelist = [Node(i) for i in range(self.n_nodes)]
        self.edgelist = [Edge(i,i) for i in range(self.n_edges)]
        self.dbalance = False     # flag to indicate if detailed balance holds

    ''' write the network to files in a format readable by Gephi '''
    def print_gephi_fmt(self):
        pass

class Analyse_coarse_ktn(object):

    def __init__(self):
        pass

    ''' set up the transition rate matrix '''
    def setup_k_mtx():
        pass

    def calc_t(self,tau):
        pass

if __name__=="__main__":

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
    print mynode1.qf

    mynode3 = Node(3)
    mynode3.node_attribs = [-0.5,4,0.25]
    myedge2 = Edge(8,8)
    myedge2.to_from_nodes = [mynode1,mynode3]
    del myedge1.to_from_nodes
    print "new ID of first IN edge of node 1:", mynode1.edgelist_in[0].edge_id
