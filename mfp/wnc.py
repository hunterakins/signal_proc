import numpy as np
from matplotlib import pyplot as plt
import numba 
from numba import jit, prange
import time

'''
Description:
Routines to compute white noise constraint processor 
Main script runs a simulation for an 11 element array and plane wave input
Array positions are perturbed and some white noise is also added to each element

Author: Hunter Akins
'''


@jit(nopython=True) 
def get_gw(d, w):
    """
    Compute Gw, the white noise gain for d
    d - numpy 2d array (column)
        the data vector
    w - numpy 2d array (colum) 
        the replica vector
    gw is defined to be norm(d w)^{2} / norm(w)^{2}
     """
    numer = w.T.conj()@d
    denom = w.T.conj()@w
    return numer/denom

@jit(nopython=True) 
def get_w_wn(outer_list, s, eps, w):
    """
    Compute the white noise gain constraint weighting vector as a function of a diagonal loading epsilon in terms of the replica w and eigendecomposition of the sample covariance
    v - numpy matrix
        eigenvectors as columns of matrix, eigenvector  of covariance matrix (guaranteed orthog.)
    s - numpy 1-d array
       eigenvalues of covariance (sorted to match up with columns of v)
    eps - float
        regularization constant
    w - numpy 2d array
        replica under consideration
    Output:
    w_wn : 
    """
    
    K_inv = get_K_inv(outer_list, s, eps)
    numer = K_inv@w
    denom = w.T.conj()@K_inv@w
    w_wn = numer/denom
    return w_wn

@jit(nopython=True)
def get_mcm_w_wn(outer_list, s, eps, E, d, E_H, d_H):
    """
    Compute thewhite noise gain constraint - MCM weighting vector as a function of a diagonal loading epsilon in terms of the replica w and eigendecomposition of the sample covariance
    v - numpy matrix
        eigenvectors as columns of matrix, eigenvector  of covariance matrix (guaranteed orthog.)
    s - numpy 1-d array
       eigenvalues of covariance (sorted to match up with columns of v)
    eps - float
        regularization constant
    E - numpy 2d array
        matrix of constraint vectors
    d - numpy 1d array
        vector of constraints
    Output:
    w_wn 
    """
    K_inv = get_K_inv(outer_list, s, eps)
    w_wn = (K_inv@E)@np.linalg.inv(E_H@K_inv@E)@(d_H)
    return w_wn

@jit(nopython=True) 
def db_down(outer_list, s, eps, w):
    """
    Get the inverse white noise gain in db for a given diagonal weighting eps
    and the eigenvectors and eigenvalues of the sample covariance v and s (respectively)
    w is the replica under consideration
    Maximum white noise gain is 1

    Input:
    outer_list - numpy 2d array
        eigenvector matrix for sample cov
    s - numpy 1d array
        eigenvalue matrix for sample cov
    eps - float
        value of diagonal loading
    w - numpy 1d array
        replica
    Output:
    db_down - float
        Number of db's below the maximum white noise value of 1
    """
    w_wn = get_w_wn(outer_list, s, eps, w)
    mag = np.square(np.linalg.norm(w_wn))#*w_wn
    # I only divide by 1 because we normalize the weight vector to 1 in Bartlett
    return 10*np.log10(1/mag)

@jit(nopython=True)
def mcm_db_down(outer_list, s, eps, E, d, E_H, d_H):
    w_wn = get_mcm_w_wn(outer_list, s, eps, E, d, E_H, d_H)
    mag = np.square(np.linalg.norm(w_wn))#*w_wn
    # I only divide by 1 because we normalize the weight vector to 1 in Bartlett
    return 10*np.log10(1/mag)


@jit(nopython=True) 
def lookup_db_down(K_inv, w):
    """ Use a precomputed K_inv (for specific eps)
        to compute white noise gain for replica w"""
    numer = K_inv@w
    denom = w.T.conj()@K_inv@w
    w_wn = numer/denom
    mag = np.square(np.linalg.norm(w_wn))#*w_wn
    return 10*np.log10(1/mag)

def find_eps(outer_list, s, db_gain, w):
    """
    Does a simple grid search for the optimal diagonal loading to give 
    the closest white noise gain to db_gain
    Input:
    v - numpy 2d array
        eigenvector matrix for sample cov
    s - numpy 1d array
        eigenvalue matrix for sample cov
    db_gain - float
        value of white noise gain in db normalized to 1
        If you want white noise gain to be fixed at 1, set db_gain = 0
    Output:
    best_eps - float
        Value of epsilon for diagonal loading
    """
    epsdb = np.linspace(-160, 6, 200)  # look for epsilon on a log scale
    eps = np.power(10.0, epsdb/10) # convert to linear scale
    vals = np.array([abs(db_down(outer_list, s, x, w)-db_gain) for x in eps]) # compute whitenoisegain for each value of epsilon
    ind = np.argmin(vals) # choose best index
    best_eps = eps[ind] # fetch the best epsilon corresponding to that
    return best_eps

@jit(nopython=True)
def mcm_find_eps_bi(outer_list, s, delta_db, E, d, E_H, d_H, tol=.01, max_num_iter=1000):
    """
    Use bisection to compute the epsilon diagonal loading value required to achieve
    a white noise gain of delta_db. For example, if you want white noise gain of 0.1, 
    delta_db = 10*np.log10(.1 /1) = -10 
    
    Input:
    v - numpy 2d array
        eigenvector matrix for sample cov
    s - numpy 1d array
        eigenvalue matrix for sample cov
    delta_db - float
        value of white noise gain in db normalized to 1
        If you want white noise gain to be fixed at 1, set delta_db = 0
    tol (optional) - float
        How close you want to be to delta_db
    max_num_iter - int
        How many cycles before you give up
    Output:

    float (could be "middle", "a_largeish_number", or "left") depending on the case 
        The "epsilon" diagonal loading used for the WNGC beamformer
    """
    left, right = np.power(10.0,-160), 1000.0 # set search bounds
    middle = np.power(10, .5*(np.log10(left) + np.log10(right))) # take geometric mean
    a_largeish_number = 1000.0 # fix the return value if delta_db = 0 is desired
    # if you want no white noise gain, epsilon should be infinite
    if delta_db == 0:
        return right
    # find out if the constraint is already satisfied (aka your white noise gain is lower than even the MVDR processor implies
    wng_left = mcm_db_down(outer_list, s, left, E, d, E_H, d_H)
    if wng_left >= delta_db: # can't get any lower
        return left
    else:
        # perform bisection to hone in on the value of epsilon
        wng_middle = mcm_db_down(outer_list,s,middle,E, d, E_H, d_H)
        num_iter = 0
        while abs(wng_middle - delta_db) > tol:
            num_iter += 1
            if wng_middle < delta_db:
                left = middle # move left bracket
            if wng_middle > delta_db:
                right = middle # move right bracket
            middle = np.power(10, .5*(np.log10(left) + np.log10(right)))  # move middle
            wng_middle = mcm_db_down(outer_list,s,middle,E, d, E_H, d_H) # compute value of white noise gain at middle
            if num_iter > max_num_iter:
                #print('Warning- Bisection did not converge in ' + str(max_num_iter) + ' iterations')
                return middle
        return middle

@jit(nopython=True) 
def find_eps_bi(outer_list, s, delta_db, w, tol=.01, max_num_iter=1000):
    """
    Use bisection to compute the epsilon diagonal loading value required to achieve
    a white noise gain of delta_db. For example, if you want white noise gain of 0.1, 
    delta_db = 10*np.log10(.1 /1) = -10 
    
    Input:
    v - numpy 2d array
        eigenvector matrix for sample cov
    s - numpy 1d array
        eigenvalue matrix for sample cov
    delta_db - float
        value of white noise gain in db normalized to 1
        If you want white noise gain to be fixed at 1, set delta_db = 0
    tol (optional) - float
        How close you want to be to delta_db
    max_num_iter - int
        How many cycles before you give up
    Output:

    float (could be "middle", "a_largeish_number", or "left") depending on the case 
        The "epsilon" diagonal loading used for the WNGC beamformer
    """
    left, right = np.power(10.0,-160), 1000.0 # set search bounds
    middle = np.power(10, .5*(np.log10(left) + np.log10(right))) # take geometric mean
    a_largeish_number = 1000.0 # fix the return value if delta_db = 0 is desired
    # if you want no white noise gain, epsilon should be infinite
    if delta_db == 0:
        return right
    # find out if the constraint is already satisfied (aka your white noise gain is lower than even the MVDR processor implies
    wng_left = db_down(outer_list, s, left, w)
    if wng_left >= delta_db: # can't get any lower
        return left
    else:
        # perform bisection to hone in on the value of epsilon
        wng_middle = db_down(outer_list,s,middle,w)
        num_iter = 0
        while abs(wng_middle - delta_db) > tol:
            num_iter += 1
            if wng_middle < delta_db:
                left = middle # move left bracket
            if wng_middle > delta_db:
                right = middle # move right bracket
            middle = np.power(10, .5*(np.log10(left) + np.log10(right)))  # move middle
            wng_middle = db_down(outer_list,s,middle,w) # compute value of white noise gain at middle
            if num_iter > max_num_iter:
                #print('Warning- Bisection did not converge in ' + str(max_num_iter) + ' iterations')
                return middle
        return middle

@jit(nopython=True) 
def int_avg(a, b):
    """
    Get 'integer' average of a and b
    """
    val = (a+b)/2
    val = numba.int64(val)
    return val
    

@jit(nopython=True) 
def lookup_eps_bi(K_inv_list, delta_db, w):
    left, right = 0, len(K_inv_list)-1
    middle = int_avg(left, right)
    a_largeish_number = 1000.0
    if delta_db == 0:
        return a_largeish_number
    K_inv_left = K_inv_list[left] 
    K_inv_middle = K_inv_list[middle]
    wng_left = lookup_db_down(K_inv_left, w)
    if wng_left >= delta_db: # can't get any lower
        return left
    else:
        # perform bisection to hone in on the value of epsilon
        wng_middle = lookup_db_down(K_inv_middle, w)
        num_iter = 0
        while (middle != left) and (middle != right):
            if wng_middle < delta_db:
                left = middle # move left bracket
            if wng_middle > delta_db:
                right = middle # move right bracket
            if wng_middle == delta_db:
                return middle
            middle = int_avg(left, right)
            K_inv_middle = K_inv_list[middle]
            wng_middle = lookup_db_down(K_inv_middle,w) # compute value of white noise gain at middle
        return numba.int64(middle)

# wnc
def pw_wnc(replicas, R_samp,delta_db):
    """
    replicas - numpy 2d array
        Each column is replica
    R_samp - numpy 2d array
        Sample cov
    delta_db - float <= 0 
        white noise gain
    Compute ambiguity surface for plane wave replicas using white noise constraint beamformer
    Output:
        output - numpy array 1d
        Ambiguity surface for replicas and sample covariance
    """
    s, v = np.linalg.eigh(R_samp) # compute eigenvectors up front
    num_guesses = replicas.shape[1] #how many replicas
    output = np.zeros(num_guesses) # ambiguity surface
    eps_list = [] # to track the diagonal loading

    """ Compute outer products up front to avoid repetitive computation
    """
    outer_list = []
    for j in range(s.size): # for each eigenvalue
        tmp = v[:,j].reshape(s.size, 1) # fetch jth eigenvector
        outer = tmp@(tmp.T.conj()) # compute outer product
        outer_list.append(outer) # append to the list
    """
    Now loop through replicas and compute beamformer output
    """
    for i in range(num_guesses): # for each replica
        w = replicas[:,i] # fetch ith replica 
        w = w / np.linalg.norm(w)  # normalize
        eps = find_eps_bi(v, s,delta_db, w) # compute epsilon
        eps_list.append(eps) # add it to the list in case you want to plot or debug
        K_inv = np.zeros(R_samp.shape, dtype=np.complex128) # initialize memory for K_inv
        """ Compute outer product inverse """
        for j in range(s.size):  # for each eigenvalue
            tmp = outer_list[j] # fetch jth outer product matrix
            K_inv += tmp/(s[j]+eps) # normalize by eigenvalue and epsilon
        """ Use the inverse to compute the weighting vector w_wnc """
        w_wnc = K_inv@w / (w.T.conj()@K_inv@w) 
        """ Plug weighting vector into quadratic form (the beamformer ''power'' output) """
        output[i] = abs(w_wnc.T.conj()@R_samp@w_wnc)
    return output

@jit(nopython=True) 
def get_eig_outers(R_samp):
    """
    Form a list of the outer products of each eigenvalue 
    in the sample covariance matrix R_samp
    Input 
    R_samp - numpy nd array
        sample cov
    Output -
    outer_list - list of numpy nd arrays
        each element is the outer product of the ith eigenvlaue    
    s - numpy array of eigenvalues
        sorted smallest to largest?
    v - numpy array of eigenvectors
        each column is a vec for the corr. eigenv
    """
    s, v = np.linalg.eigh(R_samp) # compute eigenvectors up front
    outer_list = []
    for j in range(s.size): # for each eigenvalue
        tmp = v[:,j].reshape(s.size, 1) # fetch jth eigenvector
        outer = tmp@(tmp.T.conj()) # compute outer product
        outer_list.append(outer) # append to the list
    return outer_list, s ,v 

@jit(nopython=True) 
def get_K_inv(outer_list, s, eps):
    """ Compute regularized covariance matrix inverse
    with regularization value eps and outerproduct decomposition
    of matrix (K = sum_{i} s_{i}*v_{i}@v_{i}.T
    (so inverse is sum_{i} 1/s_{i} v_{i}@v_{i}.T)
    Input - 
    outer_list - list of np arrays
        each list elem is outerproducts of eigenvectors of 
        sample covariance
    s - np array
        corresponding eigenvalues
    eps - float
        regulariz. param
    the sample cov
    """
    K_inv = np.zeros((s.size, s.size), dtype=np.complex128) # initialize memory for K_inv
    for k in range(len(outer_list)):  # for each eigenvalue
        tmp = outer_list[k] # fetch jth outer product matrix
        K_inv += tmp/(s[k]+eps) # scale by eigenvalue and epsilon
    return K_inv
   
@jit(nopython=True, parallel=True) 
def run_wnc(R_samp,replicas, delta_db):
    """
    replicas - numpy 3d array
        First axis is receiver index, second is depth, third is range
        Values are complex pressure 
    R_samp - numpy 3d array (or 2d array)
        Sample cov, last two axes are the receiver indices, first axis is time
    delta_db - float <= 0 
        white noise gain
    Compute ambiguity surface for replicas using white noise constraint beamformer
    Output:
    output - numpy array 2d
        first axis is source depths, second is source range,
        final axis is beginning time of snapshot
        values are the wnc power output.
    """
    #if len(R_samp.shape) == 2:
    #    R_samp = R_samp.reshape(R_samp.shape[0], R_samp.shape[1], 1)
    num_rcvrs = replicas.shape[0]
    num_depths = replicas.shape[1]
    num_ranges = replicas.shape[2]
    num_guesses = num_depths*num_ranges #how many replicas
    num_times = R_samp.shape[0]
    replicas = replicas.reshape(num_rcvrs, num_guesses)
    output = np.zeros((num_times, num_depths, num_ranges)) # ambiguity surface
    #eps_list = [] # to track the diagonal loading

    """
    Now loop through replicas and compute beamformer output
    """
    for i in prange(num_times):
        curr_R = R_samp[i,...]
        outer_list, s, v = get_eig_outers(curr_R)
        for j in prange(num_guesses): # for each replica
            w = replicas[:,j] # fetch ith replica 
            w = w / np.linalg.norm(w)  # normalize
        
            """ Get epsilon value for specific white noise gain """
            eps = find_eps_bi(outer_list, s,delta_db, w) # compute epsilon
            #eps_list.append(eps) # add it to the list in case you want to plot or debug
            """ Comput sample covariance regularized inverse"""
            K_inv = get_K_inv(outer_list, s, eps)

            """ Use the inverse to compute the weighting vector w_wnc """
            w_wnc = K_inv@w / (w.T.conj()@K_inv@w) 

            """ Plug weighting vector into quadratic form (the beamformer ''power'' output) """
            output[i, j//num_ranges, j%num_ranges] = abs(w_wnc.T.conj()@curr_R@w_wnc)
    return output

def lookup_run_wnc(R_samp,replicas, delta_db):
    """
    replicas - numpy 3d array
        First axis is receiver index, second is source depth, third is source range
        Values are complex pressure 
    R_samp - numpy 3d array
        Sample cov, last two axes are the receiver indices, first axis
    delta_db - float <= 0 
        white noise gain
    Compute ambiguity surface for replicas using white noise constraint beamformer
    Output:
    output - numpy array 2d
        first axis is source depths, second is source range,
        final axis is beginning time of snapshot
        values are the wnc power output.
    """
    if len(R_samp.shape) == 2:
        R_samp = R_samp.reshape(R_samp.shape[0], R_samp.shape[1], 1)
    num_rcvrs, num_depths, num_ranges = replicas.shape
    num_guesses = num_depths*num_ranges #how many replicas
    num_times = R_samp.shape[0]
    replicas = replicas.reshape(num_rcvrs, num_guesses)
    output = np.zeros((num_times, num_depths, num_ranges)) # ambiguity surface
    eps_list = [] # to track the diagonal loading

    """
    Now loop through replicas and compute beamformer output
    """
    eps_vals =  np.power(10, np.linspace(-160, 3, 1000))
    for i in prange(num_times):
        #print('Proc for time i', i)
        curr_R = R_samp[i,...]
        outer_list, s, v = get_eig_outers(curr_R)
        K_inv_list = [get_K_inv(outer_list, s, x) for x in eps_vals]
        for j in prange(num_guesses): # for each replica
            w = replicas[:,j] # fetch ith replica 
            w = w / np.linalg.norm(w)  # normalize
        
            """ Get epsilon value for specific white noise gain """
            #eps = find_eps_bi(outer_list, s,delta_db, w) # compute epsilon
            if delta_db == 0:
                eps_val = 1000
            else:
                eps_ind = lookup_eps_bi(K_inv_list, delta_db, w)
                eps_ind = numba.int64(eps_ind)
                eps_val = eps_vals[eps_ind]
            eps_list.append(eps_val) # add it to the list in case you want to plot or debug
            """ Comput sample covariance regularized inverse"""
            K_inv = get_K_inv(outer_list, s, eps_val)

            """ Use the inverse to compute the weighting vector w_wnc """
            w_wnc = K_inv@w / (w.T.conj()@K_inv@w) 

            """ Plug weighting vector into quadratic form (the beamformer ''power'' output) """
            output[i, j//num_ranges, j%num_ranges] = abs(w_wnc.T.conj()@curr_R@w_wnc)
    return output


if __name__ == '__main__':
    """
    Make some plane wave data
    """
    array_pert_var = 2 # variance of array element perturbations (gaussian)
    white_noise_var = 1 # variance of white noise on each sensor (also gaussian)
    array_pos = np.linspace(0, 100, 11) # array positoins
    lamz = 15 # 15 meter wavelength  (in z direction)
    kz = 2*np.pi/lamz # corresponding wavenumber
    num_snapshots = 40 # how many snapshots you got 

    data = np.zeros((array_pos.size, num_snapshots), dtype=np.complex128)
    K_samp = np.zeros((array_pos.size, array_pos.size), dtype=np.complex128)

    array_pos_pert = array_pos + array_pert_var*np.random.randn(11)
    plt.figure()
    plt.title("Ideal versus true array positions")
    plt.plot(array_pos_pert)
    plt.plot(array_pos)
    plt.xlabel("array element number")
    plt.ylabel("distance from origin")
    for i in range(num_snapshots):
        random_phase = np.exp(complex(0, 2*np.pi*np.random.rand()))
        pw = np.exp(complex(0, 1)*kz*array_pos_pert)
        noise = white_noise_var*np.random.randn(array_pos.size)
        d = (noise + random_phase*pw).reshape(array_pos.size, 1)
        d = d / np.linalg.norm(d)
        data[:,i] = d[:,0]
        K_samp += d@(d.T.conj())

    """
    Get sample covariance
    """
    K_samp /= num_snapshots
    s, v = np.linalg.eigh(K_samp)


    """
    Build replicas for plane wave model
    """
    k_guesses = np.linspace(0, 2*np.pi/10, 100)
    replicas = np.zeros((array_pos.size, k_guesses.size), dtype=np.complex128)
    for i in range(k_guesses.size):
        replicas[:,i] = np.exp(complex(0,1)*k_guesses[i]*array_pos)
    replicas = replicas / np.linalg.norm(replicas, axis=0)

    """
    Conventional (Bartlett) beamformer
    """
    bart = np.zeros(k_guesses.size)
    for i in range(k_guesses.size):
        w = replicas[:,i]
        w = w / np.linalg.norm(w)
        bart[i] = abs(w.T.conj() @ K_samp @ w)


    """ Minimum variance (mvdr) beamformer
    """
    mvdr = np.zeros(k_guesses.size)
    Kinv =np.linalg.inv(K_samp)
    for i in range(k_guesses.size):
        w = replicas[:,i]
        w = w / np.linalg.norm(w)
        """
        note that I don't actually need to compute w_mv to compute
        the output of the beamformer
        I just have it here because I was plotting the norm of w_mv as a function 
        of look angle
        """
        w_mv = Kinv@w/(w.T.conj()@Kinv@w) 
        mvdr[i] = 1/ abs(w.T.conj() @ Kinv @ w)
     


    delta_db = -1
    wnc = pw_wnc(replicas, K_samp, delta_db)
    plt.figure()
    plt.plot(k_guesses, bart)
    plt.plot(k_guesses, mvdr)
    plt.plot(k_guesses, wnc)
    plt.suptitle("Comparison of beamformers (true location at kz = .419)")
    plt.xlabel('Wavenumber')
    plt.legend(['bartlett', 'mvdr', 'wnc ' + str(delta_db)+ ' db'])
    plt.show()
