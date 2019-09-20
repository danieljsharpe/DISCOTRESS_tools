import numpy as np
from scipy.linalg import expm

''' Kahan summation algorithm - significantly reduces numerical error in floating point summation '''
def kahan_sum(a, axis=0):
    s = np.zeros(a.shape[:axis] + a.shape[axis+1:],dtype=np.float128)
    c = np.zeros(s.shape,dtype=np.float128)
    for i in range(a.shape[axis]):
        y = a[(slice(None),) * axis + (i,)] - c
        t = s + y
        c = (t - s) - y
        s = t.copy()
    return s

### SET PARAMS ###
n = 1000 # size of transition matrix
p = 0.2 # probability of nodes being connected
k_m = 30. # mean log rate
k_s = 5. # std dev log rate
# the transition matrix (and therefore its eigenvalues / eigenvectors) should approximately converge. We want to
# pick the smallest lag time possible giving a sensible transition matrix (diagonal elements not too close to one).
# this also seems to improve the accuracy of the matrix exponential procedure
tau = 1.E-14 # lag time for estimation of transition matrix
sum_opt = 3 # option for summation algorithm; =1 for np.sum() summation, =2 for Kahan summation
seed = 17

np.random.seed(seed)

# transition rate matrix
K = np.zeros((n,n),dtype=np.float128) # need double precision for v large rate constants


# set off-diagonal elements randomly
# Note - multiply by tau here and not later, otherwise you will multiply the floating point errors from the summation
for i in range(n):
    for j in range(n):
        if i==j: continue
        if np.random.rand() < p: # edge exists
            K[i,j] = tau*np.exp(np.random.normal(k_m,k_s))


# set diagonal elements
D = np.zeros(n,dtype=np.float128)
for i in range(n):
    if sum_opt == 1:
        K[i,i] = -np.sum(K[i,:],dtype=np.float128)
        print "K    row: %i   sum of row: %E" % (i, np.sum(K[i,:],dtype=np.float128))
    elif sum_opt == 2:
        K[i,i] = -kahan_sum(K[i,:])
        print "K    row: %i   sum of row: %E" % (i, kahan_sum(K[i,:]))
    elif sum_opt == 3: # sort numbers because it is more accurate to add numbers of similar size, and the sum is done cumulatively
        k_row = np.sort(K[i,:].copy())
        K[i,i] = -np.sum(k_row)
        print "K    row: %i   sum of row: %E" % (i, np.sum(np.sort(K[i,:].copy())))
    assert abs(np.sum(np.sort(K[i,:].copy()))) < 1.0E-08, "Error: row %i of K does not sum to 0" % i

print "transition rate matrix:\n", K

# calculate transition matrix
T = expm(K)

print "transition matrix:\n", T

for i in range(n):
    print "T    row:", i, "sum of row:", np.sum(np.sort(T[i,:].copy()))
    assert abs(np.sum(np.sort(T[i,:].copy()))-1.) < 1.0E-03, "Error: row %i of T does not sum to 1" % i # fairly relaxed error tolerance
