'''
Python script containing Node, Edge, Ktn and Coarse_ktn data structures

Daniel J. Sharpe
Sep 2019
'''

import numpy as np
from os.path import exists
from copy import copy
from copy import deepcopy

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
        self.s = None           # surprisal metric OR contribution of node to relative entropy 
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
        if (self.k_esc is None) or (self.k_in is None): self.calc_k_esc_in()
        update_edges = False
        if self.qf != None: update_edges = True
        self.qf, self.qb = vals[0], vals[1]
        self.mr = Node.calc_mr(self)
        if update_edges: # update relevant edges since old values of committor functions are now outdated
            pass

    @tpt_vals.deleter
    def tpt_vals(self):
        print "deleting tpt_vals for node", self.node_id
        self.qf, self.qb, self.mr, self.t, self.k_esc = [None]*5

    @staticmethod
    def calc_mr(node):
        if not Ktn.check_if_node(node): raise AttributeError
        return node.pi*node.qf*node.qb

    ''' calculate self-transition probability for node '''
    def calc_node_t(self):
        sum_t_out = 0.
        for edge in self.edgelist_out:
            if edge.deadts: continue
            if edge.t is None: raise AttributeError
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

    ''' return an edge that has summed flow values of two argument edges '''
    def __add__(self,other_edge):
        if not ((self.to_node.node_id==other_edge.to_node.node_id) and \
                 (self.from_node.node_id==other_edge.from_node.node_id)): raise AttributeError
        new_edge = copy(self)
        new_edge.k = np.log(np.exp(self.k)+np.exp(other_edge.k))
        if (self.t is not None) and (other_edge.t is not None): new_edge.t = self.t+other_edge.t
        if (self.j is not None) and (other_edge.j is not None): new_edge.j = self.j+other_edge.j
        if (self.f is not None) and (other_edge.f is not None): new_edge.f = self.f+other_edge.f
        if (self.fe is not None) and (other_edge.fe is not None): new_edge.fe = self.fe+other_edge.fe
        return new_edge

    ''' merge a pair of edges - only allow if start/end nodes are the same for both edges. The edge of faster rate is retained '''
    def merge_edges(self,other_edge):
        if self < other_edge: edge1, edge2 = self, other_edge
        else: edge1, edge2 = other_edge, self
        if not ((edge1.to_node==edge2.to_node) and (edge1.from_node==edge2.from_node)):
            raise AttributeError
        edge1 = edge1+edge2
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
    def flow_vals(self,dummy):
        if (self.to_node is None) or (self.from_node is None): raise AttributeError
        if self.j is None: # net flux values have not been set yet, do so now
            j1, j2 = Edge.calc_j(self)
            self.j = j1
            self.rev_edge.j = j2
        # if the committors of the TO and FROM nodes are set, can calculate TPT flux quantities now
        if (self.to_node.qf is None) or (self.from_node.qf is None): return
        if self.f is None: # reactive flux has not been set yet
            f1, f2 = Edge.calc_f(self)
            self.f = f1
            self.rev_edge.f = f2
            fe1, fe2 = Edge.calc_fe(self)
            self.fe = fe1
            self.rev_edge.fe = fe2

    @flow_vals.deleter
    def flow_vals(self):
        self.j, self.f, self.fe = None, None, None

    @staticmethod
    def calc_j(edge):
        edge1, edge2 = edge, edge.rev_edge
        if not ((Ktn.check_if_edge(edge1)) or (Ktn.check_if_edge(edge2))): raise AttributeError
        if not Ktn.check_edge_reverse(edge1,edge2): raise AttributeError
        if ((edge1.k is None) or (edge2.k is None)): raise AttributeError
        j1 = (edge1.k*edge1.from_node.pi) - (edge2.k*edge2.from_node.pi)
        j2 = -j1
        return j1, j2

    @staticmethod
    def calc_f(edge):
        edge1, edge2 = edge, edge.rev_edge
        if not ((Ktn.check_if_edge(edge1)) or (Ktn.check_if_edge(edge2))): raise AttributeError
        if not Ktn.check_edge_reverse(edge1,edge2): raise AttributeError
        f1 = np.exp(edge1.from_node.pi)*edge1.from_node.qb*np.exp(edge1.k)*edge1.to_node.qf
        f2 = np.exp(edge2.from_node.pi)*edge2.from_node.qb*np.exp(edge2.k)*edge2.to_node.qf
        return f1, f2

    @staticmethod
    def calc_fe(edge):
        edge1, edge2 = edge, edge.rev_edge
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
        self.dbalance = True      # flag to indicate if detailed balance holds
        self.A = set()            # endpoint nodes in the set A (NB A<-B)
        self.B = set()            # endpoint nodes in the set B
        self.tpt_calc_done = False # flag to indicate if (all) TPT values have been calculated for nodes and edges
        self.tau = None           # lag time at which properties (eg characteristic timescales, transition probability
                                  # matrix, etc.) of the Ktn have been calculated
        if not isinstance(self,Coarse_ktn): # do not do this for Coarse_ktn class
            self.comm_pi_vec = [-float("inf")]*self.n_comms # (log) stationary probabilities of communities
            self.comm_sz_vec = [0]*self.n_comms # number of nodes in each community

    ''' return a Ktn object that is the result of merging two TNs. The two TNs must be 'related' - that is, have the same number
        of nodes. The Ktn has averaged Node stationary probabilities and Edge transition rates and probabilities '''
    def __add__(self,other_ktn):
        if self.n_nodes!=other_ktn.n_nodes: raise AttributeError
        new_ktn = deepcopy(self)
        for i in range(new_ktn.n_nodes):
            node = new_ktn.nodelist[i]
            node_alt = other_ktn.nodelist[i]
            node.pi = np.log((np.exp(node.pi)+np.exp(node_alt.pi))/2.)
            node.t = (node.t+node_alt.t)/2.
            print i, node.t, self.nodelist[i].t, node_alt.t, (node.t+node_alt.t)/2.
            n_out_edges = len(node.edgelist_out)
            node.edgelist_out = sorted(node.edgelist_out,key=lambda x: x.to_node.node_id)
            node_alt_edgelist_out = sorted(node_alt.edgelist_out,key=lambda x: x.to_node.node_id)
            i1,i2 = 0,0
            while i1<n_out_edges and i2<len(node_alt_edgelist_out):
                curr_edge = None
                if node.edgelist_out[i1].to_node.node_id==node_alt_edgelist_out[i2].to_node.node_id:
                    node.edgelist_out[i1] = node.edgelist_out[i1]+node_alt_edgelist_out[i2]
                    curr_edge = node.edgelist_out[i1]
                    i1 += 1
                    i2 += 1
                elif node.edgelist_out[i1].to_node.node_id<node_alt_edgelist_out[i2].to_node.node_id:
                    curr_edge = node.edgelist_out[i1]
                    i1 += 1
                elif node.edgelist_out[i1].to_node.node_id>node_alt_edgelist_out[i2].to_node.node_id:
                    node.edgelist_out.append(node_alt_edgelist_out[i2])
                    curr_edge = node.edgelist_out[-1]
                    i2 += 1
                    new_ktn.n_edges += 1
                curr_edge.k = np.log(np.exp(curr_edge.k)/2.)
                curr_edge.t *= 1./2.
            if i1<n_out_edges:
                for curr_edge in node.edgelist_out[i1:n_out_edges]:
                    curr_edge.k = np.log(np.exp(curr_edge.k)/2.)
                    curr_edge.t *= 1./2.
            elif i2<len(node_alt_edgelist_out):
                for curr_edge in node_alt_edgelist_out[i2:]:
                    node.edgelist_out.append(curr_edge)
                    new_ktn.n_edges += 1
                    node.edgelist_out[-1].k = np.log(np.exp(node.edgelist_out[-1].k)/2.)
                    node.edgelist_out[-1].t *= 1./2.
            node.calc_k_esc_in()
        return new_ktn

    def construct_ktn(self,comms,conns,pi,k,t,node_ens,ts_ens):
        if ((len(node_ens)!=self.n_nodes) or (len(ts_ens)!=self.n_edges) or (len(comms)!=self.n_nodes) \
            or (len(conns)!=self.n_edges) or (len(pi)!=self.n_nodes) \
            or (len(k)!=2*self.n_edges)): raise AttributeError
        if t[0]!=None: self.tau=0. # indicates that transition probabilities are set but lag time is unknown
        for i in range(self.n_nodes):
            if comms[i] > self.n_comms-1: raise AttributeError
            self.nodelist[i].node_attribs = [node_ens[i],comms[i],pi[i]]
            self.nodelist[i].t = t[i]
        for i in range(self.n_edges):
            self.edgelist[2*i].edge_attribs = [ts_ens[i],k[2*i]]
            self.edgelist[2*i].t = t[self.n_nodes+(2*i)]
            self.edgelist[(2*i)+1].edge_attribs = [ts_ens[i],k[(2*i)+1]]
            self.edgelist[(2*i)+1].t = t[self.n_nodes+((2*i)+1)]
            # set edge connectivity
            self.edgelist[2*i].to_from_nodes = ([self.nodelist[conns[i][1]-1],self.nodelist[conns[i][0]-1]],True)
            self.edgelist[(2*i)+1].to_from_nodes = ([self.nodelist[conns[i][0]-1],self.nodelist[conns[i][1]-1]],False)
            self.edgelist[2*i].rev_edge = self.edgelist[(2*i)+1]
            self.edgelist[(2*i)+1].rev_edge = self.edgelist[2*i]
        endset_A_list = [self.nodelist[i-1] for i in Ktn.read_endpoints(endset_name="A")]
        endset_B_list = [self.nodelist[i-1] for i in Ktn.read_endpoints(endset_name="B")]
        self.A.update(endset_A_list)
        self.B.update(endset_B_list)
        self.renormalise_pi(mode=0) # check node stationary probabilities are normalised
        self.get_comm_stat_probs() # get stationary probabilities and no's of nodes for communities
        for node in self.nodelist:
            node.calc_k_esc_in()
        for edge in self.edgelist:
            edge.flow_vals

    ''' read (at least) node connectivity, stationary distribution and transition rates from files '''
    @staticmethod
    def read_ktn_info(n_nodes,n_edges,ktn_id=""):
        comms = Ktn.read_single_col("communities"+ktn_id+".dat",n_nodes)
        conns = Ktn.read_double_col("ts_conns"+ktn_id+".dat",n_edges)
        pi = Ktn.read_single_col("stat_prob"+ktn_id+".dat",n_nodes,fmt="float")
        k = Ktn.read_single_col("ts_weights"+ktn_id+".dat",2*n_edges,fmt="float")
        t = Ktn.read_single_col("ts_probs"+ktn_id+".dat",n_nodes+(2*n_edges),fmt="float")
        node_ens = Ktn.read_single_col("node_ens"+ktn_id+".dat",n_nodes,fmt="float") # optional
        ts_ens = Ktn.read_single_col("ts_ens"+ktn_id+".dat",n_edges,fmt="float") # optional
        return comms, conns, pi, k, t, node_ens, ts_ens

    ''' read forward and backward committor functions from files and update Ktn data structure '''
    def read_committors(self,ktn_id=""):
        if not exists("qf"+ktn_id+".dat") or not exists("qb"+ktn_id+".dat"): raise RuntimeError
        qf = Ktn.read_single_col("qf"+ktn_id+".dat",self.n_nodes,fmt="float")
        qb = Ktn.read_single_col("qb"+ktn_id+".dat",self.n_nodes,fmt="float")
        self.update_all_tpt_vals(qf,qb)

    ''' write the network to files in a format readable by Gephi '''
    def print_gephi_fmt(self,fmt="csv",mode=0,evec_idx=0):
        if exists("ktn_nodes."+fmt) or exists("ktn_edges."+fmt): raise RuntimeError
        ktn_nodes_f = open("ktn_nodes."+fmt,"w")
        ktn_edges_f = open("ktn_edges."+fmt,"w")
        if fmt=="csv":
            ktn_nodes_f.write("Id,Label,Energy,Community,pi,mr,qf,qb,t,s,evec")
            if isinstance(self,Coarse_ktn): ktn_nodes_f.write(",evec_err\n")
            else: ktn_nodes_f.write("\n")
            ktn_edges_f.write("Source,Target,Weight,Type,Energy,t,j,f,fe\n")
        if fmt=="csv":
            for node in self.nodelist:
                ktn_nodes_f.write(str(node.node_id)+","+str(node.node_id)+","+str(node.node_en)+","+\
                    str(node.comm_id)+","+str(np.exp(node.pi))+","+str(node.mr)+","+str(node.qf)+","+str(node.qb)+\
                    ","+str(node.t)+","+str(node.s)+","+str(node.evec[evec_idx]))
                if self.__class__.__name__ == "Coarse_ktn":
                    ktn_nodes_f.write(","+str(node.evec_err[evec_idx]))
                ktn_nodes_f.write("\n")
        ktn_nodes_f.close()
        if fmt=="csv" and mode==0: # directed edges (direction determined by net flux)
            for edge in self.edgelist:
                if edge.deadts: continue
                edge_type = "directed"
                if mode==0 and edge.fe==0.: continue # directed edges, direction determined by net reactive flux
                if mode==1 and edge.j<0.: continue # directed edges, direction determined by net flux
                if mode==2: edge_type = "undirected" # print all edges, edges are undirected
                ktn_edges_f.write(str(edge.from_node.node_id)+","+str(edge.to_node.node_id)+","+\
                    str(np.exp(edge.k))+","+edge_type+","+str(edge.ts_en)+","+str(edge.t)+","+str(edge.j)+","+\
                    str(edge.f)+","+str(edge.fe)+"\n")
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
    def read_endpoints(endset_name="A"):
        endset_fname = "min."+endset_name
        if not exists(endset_fname): raise RuntimeError
        endset_f = open(endset_fname,"r")
        endset = [int(line) for line in endset_f.readlines()[1:]]
        endset_f.close()
        return endset

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

        ''' update committor function values in Node data structures, and set other TPT values in the Node's and
            Edge's of the Ktn data structure (eg reactive fluxes etc) '''
    def update_all_tpt_vals(self,qf,qb):
        for node in self.nodelist:
            node.tpt_vals = [qf[node.node_id-1], qb[node.node_id-1]]
        for edge in self.edgelist:
            if edge.deadts: continue
            edge.flow_vals = 0
        self.tpt_calc_done = True

    def get_comm_stat_probs(self):
        self.comm_pi_vec = [-float("inf")]*self.n_comms
        for node in self.nodelist:
            self.comm_pi_vec[node.comm_id] = np.log(np.exp(self.comm_pi_vec[node.comm_id]) + \
                np.exp(node.pi))
            self.comm_sz_vec[node.comm_id] += 1
        assert abs(sum([np.exp(comm_pi) for comm_pi in self.comm_pi_vec])-1.) < 1.E-10

    def construct_coarse_ktn(self):
        coarse_ktn = Coarse_ktn(self)
        return coarse_ktn

    ''' dump relevant node information to files: eigenvectors, eigenvector errors, communities. Also write
        committor functions and reactive trajectory current if calculated. If called
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
                    evec_coarse_f.write("%1.12f\n" % (coarse_network.nodelist[node.comm_id].evec[i] / \
                                        float(full_network.comm_sz_vec[node.comm_id])))
                    evec_err_f.write("%1.12f\n" % node.evec_err[i])
            evec_full_f.close()
            if coarse_network is not None:
                evec_coarse_f.close()
                evec_err_f.close()
        if exists("communities_new.dat"): raise RuntimeError
        with open("communities_new.dat","w") as comms_new_f:
            for node in full_network.nodelist:
                comms_new_f.write(str(node.comm_id)+"\n")
        for [network, nw_str] in zip([full_network,coarse_network],["full","coarse"]):
            if network is None: continue
            if not network.tpt_calc_done: continue
            if (exists("qf."+nw_str+".dat") or exists("qb."+nw_str+".dat") or \
                exists("mr."+nw_str+".dat")): raise RuntimeError
            qf_f = open("qf."+nw_str+".dat","w")
            qb_f = open("qb."+nw_str+".dat","w")
            mr_f = open("mr."+nw_str+".dat","w")
            for node in network.nodelist:
                qf_f.write("%1.12f\n" % node.qf)
                qb_f.write("%1.12f\n" % node.qb)
                mr_f.write("%1.12f\n" % node.mr)
            qf_f.close()
            qb_f.close()
            mr_f.close()

    ''' dump relevant edge information to files '''
    def write_edges_info(self):
        full_network, coarse_network = self, None
        if isinstance(self,Coarse_ktn): coarse_network, full_network = self, self.parent_ktn
        full_tprobs, coarse_tprobs = False, False # flags indicate if transition probabilities have been calculated
        for [network, nw_str] in zip([full_network,coarse_network],["full","coarse"]):
            if (exists("reactive_flux."+nw_str+".dat") or exists("net_reactive_flux."+nw_str+".dat") or \
                exists("trans_probs."+nw_str+".dat")): raise RuntimeError
            if network.tau is not None:
                trans_probs_f = open("trans_probs."+nw_str+".dat","w")
                for node in network.nodelist:
                    trans_probs_f.write("%i %i   %1.12f\n" % (node.node_id,node.node_id,node.t))
            if network.tpt_calc_done:
                reac_flux_f = open("reactive_flux."+nw_str+".dat","w")
                net_reac_flux_f = open("net_reactive_flux."+nw_str+".dat","w")
            for edge in network.edgelist:
                if edge.deadts: continue
                if network.tau is not None:
                    trans_probs_f.write("%i %i   %1.12f\n" % (edge.from_node.node_id,edge.to_node.node_id,edge.t))
                if network.tpt_calc_done:
                    reac_flux_f.write("%i %i   %1.12f\n" % (edge.from_node.node_id,edge.to_node.node_id,edge.f))
                    net_reac_flux_f.write("%i %i   %1.12f\n" % (edge.from_node.node_id,edge.to_node.node_id,edge.fe))
            if network.tau is not None:
                trans_probs_f.close()
            if network.tpt_calc_done:
                reac_flux_f.close()
                net_reac_flux_f.close()

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
        for edge in self.edgelist:
            edge.flow_vals
        for node in self.parent_ktn.A:
            self.A.add(self.nodelist[self.parent_ktn.nodelist[node.node_id-1].comm_id])
        for node in self.parent_ktn.B:
            if node in self.A: raise RuntimeError # the macrostates A and B overlap
            self.B.add(self.nodelist[self.parent_ktn.nodelist[node.node_id-1].comm_id])

    ''' calculate errors in the n dominant eigenvectors of the transition matrix for the parent KTN compared
        to the coarsened ktn '''
    def get_evec_errors(self,n):
        for node in self.parent_ktn.nodelist:
            node.evec_err = [0.]*n
            for j in range(n):
                node.evec_err[j] = abs(node.evec[j]-(self.nodelist[node.comm_id].evec[j] \
                                       /float(self.parent_ktn.comm_sz_vec[node.comm_id])))
