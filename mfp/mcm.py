import numpy as np
from matplotlib import pyplot as plt
from signal_proc.mfp.wnc import get_eig_outers, mcm_find_eps_bi, get_K_inv, get_mcm_w_wn

"""
Description:
Implementation of the Multiple Constraint beamformer
(Schmidt, Baggeroer, Kuperman, and Scheer (SBKS) 1990)

Date:
11/5/2020

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

def form_constraint_mat(rep_list, look_ind=0):
    """
    Form the constraint matrix E and contraint vector
    d for the list of replicas
    rep_list - list of numpy arrays
        each array is a set of replicas 
        the replicas are np nd arrays
        first axis is receiver index, second is depth
        third is range
        the list is formed such that 
        rep_list[i][:,k,l] is the ith column of 
        the constraint matrix E
        For example, to do MCM with neighboring replica
        constraints, you would take the replica matrix, r_mat, for 
        the grid points you've selected and form rep_list as

        rep_list = [replicas[:, 1:-1,1:-1], replicas[:, :-2, :-2], replicas[:, 2:, :-2], replicas[:, :-2, 2:], replicas[:, 2:, 2:]]
        rep_list[look_ind] is the look direction
    """
    num_constraint_pts = len(rep_list)
    look_replica = rep_list[look_ind]
    num_rcvrs = look_replica.shape[0]
    num_look_directions = look_replica.shape[1]*look_replica.shape[2]
    print('num look directions', num_look_directions)
    E_mat = np.zeros((num_look_directions, num_rcvrs, num_constraint_pts), dtype=np.complex128)
    d_vecs = np.zeros((num_look_directions, num_constraint_pts), dtype=np.complex128)
    look_replica = look_replica.reshape(num_rcvrs, num_look_directions)
    for i in range(num_constraint_pts):
        ith_rep = rep_list[i]
        ith_rep = ith_rep.reshape(num_rcvrs, num_look_directions)
        rep_list[i] = ith_rep
        d_vecs[:,i] = np.einsum('ij,ji->i', look_replica.T.conj(), ith_rep)
    for i in range(num_constraint_pts):
        E_mat[:,:,i] = rep_list[i].T
    return E_mat, d_vecs
    
def run_mcm(R_samp, rep_list, look_ind=0):
    """
    Following notation in SBKS, R_samp is K,
    E is the matrix of replicas for the constraint
    points
    Input - 
    R_samp - numpy nd array
        if dim is three, then the last axis is time
    rep_list - list of numpy arrays
        See form_constraint_mat
    look_ind - integer
        rep_list[look_ind] is the look direction
    """
    
    if len(R_samp.shape) == 2:
        R_samp = R_samp.reshape(R_samp.shape[0], R_samp.shape[1], 1)
    num_times = R_samp.shape[-1]
    num_depths, num_ranges = rep_list[0].shape[1], rep_list[0].shape[2]
    E, d_vecs = form_constraint_mat(rep_list, look_ind=look_ind) 
    num_look_directions = E.shape[0]
    output = np.zeros((num_depths, num_ranges, num_times))
    for i in range(num_times):
        K = R_samp[:,:,i]
        K_inv = np.linalg.inv(K)
        for j in range(num_look_directions):
            curr_E = E[j,:,:]
            curr_E = np.matrix(curr_E)
            curr_d = d_vecs[j, :]
            curr_d = curr_d.reshape(1,curr_d.size)
            pow_val = curr_d@np.linalg.inv(curr_E.H@K_inv@curr_E)@(curr_d.T.conj())
            output[j//num_ranges, j%num_ranges,i] = abs(pow_val)
    return output

def run_wnc_mcm(R_samp, rep_list, delta_db, look_ind=0):
    """
    Following notation in SBKS, R_samp is K,
    E is the matrix of replicas for the constraint
    points
    Input - 
    R_samp - numpy nd array
        if dim is three, then the last axis is time
    rep_list - list of numpy arrays
        See form_constraint_mat
    look_ind - integer
        rep_list[look_ind] is the look direction
    """
    
    if len(R_samp.shape) == 2:
        R_samp = R_samp.reshape(R_samp.shape[0], R_samp.shape[1], 1)
    num_times = R_samp.shape[-1]
    num_depths, num_ranges = rep_list[0].shape[1], rep_list[0].shape[2]
    E, d_vecs = form_constraint_mat(rep_list, look_ind=look_ind) 
    num_look_directions = E.shape[0]
    output = np.zeros((num_depths, num_ranges, num_times))
    for i in range(num_times):
        K = R_samp[:,:,i]
        #K_inv = np.linalg.inv(K)
        outer_list, s, v = get_eig_outers(K)
        for j in range(num_look_directions):
            curr_E = E[j,:,:]
            curr_E = np.matrix(curr_E)
            curr_d = d_vecs[j, :]
            curr_d = curr_d.reshape(1,curr_d.size)
            eps = mcm_find_eps_bi(outer_list, s, delta_db, curr_E, curr_d)
            w_mcm = get_mcm_w_wn(outer_list, s, eps, curr_E, curr_d)
            pow_val = abs(w_mcm.T.conj()@K@w_mcm)
            output[j//num_ranges, j%num_ranges,i] = pow_val
    return output
