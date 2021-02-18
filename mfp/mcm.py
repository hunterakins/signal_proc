import numpy as np
from matplotlib import pyplot as plt
from signal_proc.mfp.wnc import get_eig_outers, mcm_find_eps_bi, get_K_inv, get_mcm_w_wn
import numba
from numba import jit,prange
"""
Description:
Implementation of the Multiple Constraint beamformer
(Schmidt, Baggeroer, Kuperman, and Scheer (SBKS) 1990)

Date:
11/5/2020

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

@jit(nopython=True, parallel=True)
def form_constraint_mat(rep_arr, look_ind=0):
    """
    Form the constraint matrix E and contraint vector
    d for the list of replicas
    rep_arr - numpy arrays 4d array
        first axis gives the replica set for a constraint point
        the replicas are np nd arrays
        second axis is receiver index,third is depth
        fourth is range
        the list is formed such that 
        rep_arr[i,:,k,l] is the ith column of 
        the constraint matrix E
        For example, to do MCM with neighboring replica
        constraints, you would take the replica matrix, r_mat, for 
        the grid points you've selected and form rep_list as

        rep_list = [replicas[:, 1:-1,1:-1], replicas[:, :-2, :-2], replicas[:, 2:, :-2], replicas[:, :-2, 2:], replicas[:, 2:, 2:]]
        rep_arr[look_ind,...] is the look direction
    """
    num_constraint_pts = rep_arr.shape[0]
    look_replica = rep_arr[look_ind,...]
    num_rcvrs = look_replica.shape[0]
    num_look_directions = look_replica.shape[1]*look_replica.shape[2]
    print('num look directions', num_look_directions)
    E_mat = np.zeros((num_look_directions, num_rcvrs, num_constraint_pts), dtype=np.complex128)
    d_vecs = np.zeros((num_look_directions, num_constraint_pts), dtype=np.complex128)
    #print(numba.typeof(look_replica))
    look_replica = look_replica.reshape(num_rcvrs, num_look_directions)
    for i in prange(num_constraint_pts):
        ith_rep = rep_arr[i,...]
        ith_rep = ith_rep.reshape(num_rcvrs, num_look_directions)
        #d_vecs[:,i] = np.einsum('ij,ji->i', look_replica.T.conj(), ith_rep)
        d_vecs[:,i] = np.sum(look_replica.conj() * ith_rep, axis=0)
        E_mat[:,:,i] = ith_rep.T
    return E_mat, d_vecs

@jit(nopython=True, parallel=True)
def run_mcm(R_samp, rep_arr, look_ind=0):
    """
    Following notation in SBKS, R_samp is K,
    E is the matrix of replicas for the constraint
    points
    Input - 
    R_samp - numpy nd array
        dim is three, first axis is time
    rep_arr - numpy array
        See form_constraint_mat
    look_ind - integer
        rep_arr[look_ind,:,:,:] is the look direction
    """
    
    num_times = R_samp.shape[0]
    num_depths, num_ranges = rep_arr.shape[2], rep_arr.shape[3]
    E, d_vecs = form_constraint_mat(rep_arr, look_ind=look_ind) 
    num_look_directions = E.shape[0]
    print(num_look_directions)
    output = np.zeros((num_times, num_depths, num_ranges))
    for i in prange(num_times):
        K = R_samp[i,...]
        K_inv = np.linalg.inv(K)
        for j in prange(num_look_directions):
            curr_E = E[j,...]
            curr_E_T = np.ascontiguousarray(curr_E.T)
            curr_d = d_vecs[j, :]
            curr_d = curr_d.reshape(1,curr_d.size)
            curr_d_T = np.ascontiguousarray(curr_d.T)
            middle_term = curr_E.T.conj()@K_inv@curr_E
            pow_val = curr_d@np.linalg.inv(curr_E_T.conj()@K_inv@curr_E)@(curr_d_T.conj())
            pow_val = pow_val[0,0]
            output[i, j//num_ranges, j%num_ranges] = abs(pow_val)
    return output

@jit(nopython=True, parallel=True)
def run_wnc_mcm(R_samp, rep_arr, delta_db, look_ind=0):
    """
    Following notation in SBKS, R_samp is K,
    E is the matrix of replicas for the constraint
    points
    Input - 
    R_samp - numpy nd array
        dim is three, the first axis is time
    rep_arr - numpy array
        See form_constraint_mat
    look_ind - integer
        rep_arr[look_ind,...] is the look direction

    Notes 
    Since R_samp is a covariacne matrix, it should be semi-pos  
    def
    Therefore the quadratic form should be real (and >= 0), so instead of
    takins abs i just take the real part
    

    """
    
    num_times = R_samp.shape[0]
    num_depths, num_ranges = rep_arr.shape[2], rep_arr.shape[3]
    E, d_vecs = form_constraint_mat(rep_arr, look_ind=look_ind) 
    num_look_directions = E.shape[0]
    output = np.zeros((num_times, num_depths, num_ranges))
    eps_list = []
    for i in prange(num_times):
        K = R_samp[i,...]
        #K_inv = np.linalg.inv(K)
        outer_list, s, v = get_eig_outers(K)
        for j in prange(num_look_directions):
            curr_E = E[j,...]
            curr_E_H = np.ascontiguousarray(curr_E.T.conj())
            curr_d = d_vecs[j, :]
            curr_d = curr_d.reshape(1,curr_d.size)
            curr_d_H = np.ascontiguousarray(curr_d.T.conj())
            eps = mcm_find_eps_bi(outer_list, s, delta_db, curr_E, curr_d, curr_E_H, curr_d_H)
            eps_list.append(eps)
            w_mcm = get_mcm_w_wn(outer_list, s, eps, curr_E, curr_d, curr_E_H, curr_d_H)
            w_mcm_H = np.ascontiguousarray(w_mcm.T.conj())
            pow_val = w_mcm_H@K@w_mcm
            pow_val = pow_val[0,0]
            output[i, j//num_ranges, j%num_ranges] = abs(pow_val)
    return output
