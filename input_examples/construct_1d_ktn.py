'''
Python script to construct a 1D kinetic transition network (cf Manhart, Kion-Crosby & Morozov JCP 2015)
'''

import numpy as np

n_nodes = 10 # length of 1d network
beta = 10. # inverse temperature

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
