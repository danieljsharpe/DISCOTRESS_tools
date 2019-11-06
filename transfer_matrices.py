'''
Algorithm of Manhart & Morozov (Phys Rev Lett 2013, J Chem Phys 2015, PNAS 2015) to calculate moments of the distributions of path statistics

Daniel J. Sharpe
'''

from ktn_structures import Node, Edge, Ktn, Coarse_ktn
from ktn_analysis_methods import Analyse_ktn
from scipy.sparse import csr_matrix

''' set up the initial transfer matrices, transfer vectors, cumulative vectors and other objects needed by the algorithm
    of Manhart & Morozov '''
def setup_transfer_matrices(ktn):
    pass


''' Main function for the algorithm of Manhart & Morozov '''
def manhart_morozov(ktn):
    Analyse_coarse_ktn.calc_tbranch(ktn) # get the transition matrix as the matrix of branching probabilities (jump matrix)
