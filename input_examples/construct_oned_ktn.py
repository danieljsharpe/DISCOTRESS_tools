'''
Python script to construct a 1D kinetic transition network (cf Manhart, Kion-Crosby & Morozov JCP 2015)

Daniel J. Sharpe
Nov 2019
'''

import numpy as np

n_nodes = 10 # length of 1d network
beta = 10. # inverse temperature
print_ps = True # also write data files in pathsample format Y/N

with open("ts_conns_1d.dat","w") as ts_conns_f:
    for i in range(1,n_nodes):
        ts_conns_f.write("%i %i\n" % (i,i+1))

# Metropolis transition rates given inverse temperature, as logs
with open("ts_weights_1d.dat","w") as ts_wts_f:
    for i in range(1,n_nodes):
        ts_wts_f.write("%1.12f\n%1.12f\n" % (0.,-beta/(n_nodes-1)))

# uniform stationary probabilities, as logs
with open("stat_prob_1d.dat","w") as stat_prob_f:
    for i in range(n_nodes):
        stat_prob_f.write("%1.12f\n" % np.log(1./n_nodes))

# communities file (A, I and B sets)
with open("communities_1d.dat","w") as comms_f:
    for i in range(n_nodes):
        if i==0: comm_id=0
        elif i==n_nodes-1: comm_id=2
        else: comm_id=1
        comms_f.write("%i\n" % comm_id)

if not print_ps: quit()

# Note: in writing a 1D network in PATHSAMPLE format, we make the "ts energy" between any two nodes
# be drawn from a Gaussian distribution. This ensures that there are no ties.
# the energies of nodes are given by a linearly decreasing potential. The drawn energy barrier value
# is added to the mean energy of the nodes it connects
# This is not the same as the above for non-PATHSAMPLE format
mean = 0.4 # ensure that the mean and std_dev of the Gaussian are defined that samples drawn are >> delta_en/2.
std_dev = 0.04
np.random.seed(19)

delta_en = 1./float(n_nodes-1)

with open("min.data.1d","w") as md_f:
    for i in range(n_nodes): # NB the 1's are just dummy values
        node_en = 1.-(i*delta_en)
        md_f.write("%1.5f  %i  %i  %i  %i  %i\n" % (node_en,1,1,1,1,1))

with open("ts.data.1d","w") as tsd_f:
    for i in range(1,n_nodes):
        ts_en = 1.-(i*delta_en)-(delta_en/2.)
        ts_en += np.random.normal(mean,std_dev)
        tsd_f.write("%1.5f  %i  %i  %i  %i  %i  %i  %i\n" % (ts_en,0,1,i,i+1,1,1,1))
