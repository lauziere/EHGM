
import numpy as np
import pandas as pd
import os

from util import *

# QAP
def get_pair_arrays(key_list, array_list):
    
    num_pairs = int(array_list[0].shape[1]/2)
    lefts = np.arange(2*num_pairs-2,-2,-2)
    rights = np.arange(2*num_pairs-1,-1,-2)

    # print(num_pairs, lefts, rights)

    num_samples = len(key_list) 
    sample_sizes = np.empty(num_samples, dtype='int')
    
    for j in range(num_samples):
        
        sample_sizes[j] = len(key_list[j])

    pair_arrays = []
    
    for j in range(num_samples):
        
        n = sample_sizes[j]
        arrays = array_list[j]
        pair_arrays_temp = np.empty((n, num_pairs))

        for k in range(n):

            vol = arrays[k]
            left_side = vol[lefts]
            right_side = vol[rights]
            dists = np.array([np.linalg.norm(right_side[z]-left_side[z]) for z in range(num_pairs)])

            pair_arrays_temp[k] = dists

        pair_arrays.append(pair_arrays_temp)

    return pair_arrays    

def get_side_lens(key_list, array_list):

    num_pairs = int(array_list[0].shape[1]/2)
    lefts = np.arange(2*num_pairs-2,-2,-2)
    rights = np.arange(2*num_pairs-1,-1,-2)

    num_samples = len(key_list) 
    sample_sizes = np.empty(num_samples, dtype='int')
    
    for j in range(num_samples):
        
        sample_sizes[j] = len(key_list[j])
    
    left_sides = []
    right_sides = []

    for i in range(num_samples):
        
        data_set = array_list[i]
        n_sample = sample_sizes[i]
        
        lens_sample = np.empty((n_sample, 2, num_pairs-1))
        
        for j in range(n_sample):
        
            vol = data_set[j]

            left_side = vol[lefts]
            right_side = vol[rights]

            each_side = [left_side, right_side]

            for k in range(2):

                this_side = each_side[k]

                side_lens = [np.linalg.norm(this_side[z+1] - this_side[z]) for z in range(num_pairs-1)]

                lens_sample[j, k] = side_lens

        left_side_sample = lens_sample[:,0,:]
        right_side_sample = lens_sample[:,1,:]

        left_sides.append(left_side_sample)
        right_sides.append(right_side_sample)
        
    return [left_sides, right_sides]

# Pairs
# Degree 4 features:
def get_pair_ratio_arrays(key_list, array_list):
    
    num_pairs = int(array_list[0].shape[1]/2)
    lefts = np.arange(2*num_pairs-2,-2,-2)
    rights = np.arange(2*num_pairs-1,-1,-2)

    num_samples = len(key_list) 
    sample_sizes = np.empty(num_samples, dtype='int')
    
    for j in range(num_samples):
        
        sample_sizes[j] = len(key_list[j])
        
    pair_arrays = []
    
    for j in range(num_samples):
        
        n = sample_sizes[j]
        arrays = array_list[j]
        pair_arrays_temp = np.empty((n, num_pairs-1))

        for k in range(n):

            vol = arrays[k]
            left_side = vol[lefts]
            right_side = vol[rights]
            dists = np.array([np.linalg.norm(right_side[z+1]-left_side[z+1])/np.linalg.norm(right_side[z]-left_side[z]) for z in range(num_pairs-1)])

            pair_arrays_temp[k] = dists

        pair_arrays.append(pair_arrays_temp)

    return pair_arrays

def get_cos_sims(key_list, array_list):
        
    num_pairs = int(array_list[0].shape[1]/2)
    lefts = np.arange(2*num_pairs-2,-2,-2)
    rights = np.arange(2*num_pairs-1,-1,-2)

    num_samples = len(key_list) 
    sample_sizes = np.empty(num_samples, dtype='int')
    
    for j in range(num_samples):
        
        sample_sizes[j] = len(key_list[j])
    
    cos_sims = []
     
    for i in range(num_samples):
        
        data_set = array_list[i]
        n_sample = sample_sizes[i]
        
        cos_sims_sample = np.empty((n_sample, num_pairs-1))
        
        for j in range(n_sample):

            vol = data_set[j]

            left_side = vol[lefts]
            right_side = vol[rights]

            cos_sims_temp = np.empty((num_pairs-1))

            for k in range(num_pairs-1):

                l_p_left = left_side[k]
                c_p_left = left_side[k+1]

                l_p_right = right_side[k]
                c_p_right = right_side[k+1]

                left_ray = c_p_left - l_p_left
                right_ray = c_p_right - l_p_right

                left_ray_mag = np.linalg.norm(left_ray)
                right_ray_mag = np.linalg.norm(right_ray)

                ray_dot = np.dot(left_ray, right_ray)
                ray_dot_norm = ray_dot/(left_ray_mag*right_ray_mag)

                cos_sims_temp[k] = ray_dot_norm

            cos_sims_sample[j] = cos_sims_temp

        cos_sims.append(cos_sims_sample)

    return cos_sims

def get_axial_twists(key_list, array_list):

    num_pairs = int(array_list[0].shape[1]/2)
    lefts = np.arange(2*num_pairs-2,-2,-2)
    rights = np.arange(2*num_pairs-1,-1,-2)

    num_samples = len(key_list) 
    sample_sizes = np.empty(num_samples, dtype='int')
    
    for j in range(num_samples):
        
        sample_sizes[j] = len(key_list[j])

    twists = []
     
    for i in range(num_samples):
        
        data_set = array_list[i]
        n_sample = sample_sizes[i]
        
        twists_sample = np.empty((n_sample, num_pairs-1))
        
        for j in range(n_sample):

            vol = data_set[j]

            left_side = vol[lefts]
            right_side = vol[rights]

            twist_temp = np.empty((num_pairs-1))

            for k in range(num_pairs-1):

                lp_left = left_side[k]
                cp_left = left_side[k+1]

                lp_right = right_side[k]
                cp_right = right_side[k+1]

                b1 = lp_left - cp_left
                b1_u = b1 / np.linalg.norm(b1)

                b2 = lp_right - lp_left
                b2_u = b2 / np.linalg.norm(b2)

                b3 = cp_right - lp_right
                b3_u = b3 / np.linalg.norm(b3)

                b4 = cp_left - cp_right
                b4_u = b4 / np.linalg.norm(b4)

                n_12 = np.cross(b1, b2)
                n_23 = np.cross(b2, b3)
                n_34 = np.cross(b3, b4)

                in1 = np.dot(np.cross(n_23, n_34), b3_u)
                in2 = np.dot(n_23, n_34)

                angle = np.arctan2(in1, in2)

                twist_temp[k] = angle/np.pi

            twists_sample[j] = twist_temp

        twists.append(twists_sample)

    return twists

def get_lateral_axial_twists(key_list, array_list):

    num_pairs = int(array_list[0].shape[1]/2)
    lefts = np.arange(2*num_pairs-2,-2,-2)
    rights = np.arange(2*num_pairs-1,-1,-2)
    num_samples = len(key_list) 
    sample_sizes = np.empty(num_samples, dtype='int')
    
    for j in range(num_samples):
        
        sample_sizes[j] = len(key_list[j])
    
    twists = []
     
    for i in range(num_samples):
        
        data_set = array_list[i]
        n_sample = sample_sizes[i]
        
        twists_sample = np.empty((n_sample, num_pairs-1))
        
        for j in range(n_sample):

            vol = data_set[j]

            left_side = vol[lefts]
            right_side = vol[rights]

            twist_temp = np.empty((num_pairs-1))

            for k in range(num_pairs-1):

                lp_left = left_side[k]
                cp_left = left_side[k+1]

                lp_right = right_side[k]
                cp_right = right_side[k+1]

                b1 = cp_left - lp_left 
                b1_u = b1 / np.linalg.norm(b1)
                
                b2 = lp_left - lp_right
                b2_u = b2 / np.linalg.norm(b2)
                
                b3 = lp_right - cp_right
                b3_u = b3 / np.linalg.norm(b3)
                
                b4 = cp_right - cp_left 
                b4_u = b4 / np.linalg.norm(b4)
                
                n_12 = np.cross(b1, b2)
                n_23 = np.cross(b2, b3)
                n_34 = np.cross(b3, b4)
                
                in1 = np.dot(np.cross(n_12, n_23), b2_u)
                in2 = np.dot(n_12, n_23)
                
                angle = np.arctan2(in1, in2)

                twist_temp[k] = angle/np.pi

            twists_sample[j] = twist_temp

        twists.append(twists_sample)

    return twists

# Degree 6 features:
def get_bend_angles(key_list, array_list):
    
    num_pairs = int(array_list[0].shape[1]/2)
    lefts = np.arange(2*num_pairs-2,-2,-2)
    rights = np.arange(2*num_pairs-1,-1,-2)

    num_samples = len(key_list) 
    sample_sizes = np.empty(num_samples, dtype='int')
    
    for j in range(num_samples):
        
        sample_sizes[j] = len(key_list[j])
    
    n_total = int(sample_sizes.sum())
    
    # Set up objects
    midpoints = []
    midpoint_distances = []
    angles = []
    angle_sums = []
    
    # Get midpoints for samples
    for j in range(num_samples):
        
        n = sample_sizes[j]
        arrays = array_list[j]
        midpoints_temp = np.empty((n, num_pairs, 3))

        for k in range(n):

            vol = arrays[k]
            left_side = vol[lefts]
            right_side = vol[rights]
            mp = np.array([.5*(left_side[j]+right_side[j]) for j in range(num_pairs)])

            midpoints_temp[k] = mp

        midpoints.append(midpoints_temp)
    
    # Get midpoint distances, angles, angle sums for samples
    for j in range(num_samples):
        
        n = sample_sizes[j]
        arrays = array_list[j]
        
        mps = midpoints[j]
        
        midpoint_distances_temp = np.empty((n, num_pairs-1))
        angles_temp = np.empty((n, num_pairs-2))
        angle_sums_temp = np.empty(n)
    
        for k in range(n):
            
            vol = arrays[k]
            mp = mps[k]
            
            midpoint_distances_temp[k] = np.array([np.linalg.norm(mp[z+1]-mp[z]) for z in range(num_pairs-1)])
            ANGLES_k = np.empty(0)

            for l in range(1, num_pairs-1):

                # j=1: j-1: head, j: pair 2, j+1: pair 3
                # j=2: j-1: pair 2, j: pair 3, j+1: pair 4
                # ...
                # j=9: j-1: pair 8, j: pair 9, j+1: tail

                n_p = mp[l+1]
                c_p = mp[l]
                l_p = mp[l-1]

                r = np.dot(l_p-c_p, n_p-c_p)
                r_norm = r / (np.linalg.norm(l_p-c_p) * np.linalg.norm(n_p-c_p))

                angle = (180/np.pi)*np.arccos(r_norm)
                ANGLES_k = np.append(ANGLES_k, angle)

            sum_angles = ANGLES_k.sum()
            angle_sums_temp[k] = sum_angles
            angles_temp[k] = ANGLES_k

        midpoint_distances.append(midpoint_distances_temp)
        angles.append(angles_temp)
        angle_sums.append(angle_sums_temp)
    
    return midpoints, midpoint_distances, angles, angle_sums

def get_plane_intersection_angles(key_list, array_list):

    num_pairs = int(array_list[0].shape[1]/2)
    lefts = np.arange(2*num_pairs-2,-2,-2)
    rights = np.arange(2*num_pairs-1,-1,-2)

    num_samples = len(key_list) 
    sample_sizes = np.empty(num_samples, dtype='int')
    
    for j in range(num_samples):
        
        sample_sizes[j] = len(key_list[j])
    
    n_total = int(sample_sizes.sum())
    
    # Set up objects
    midpoints = []
    angles = []

    # Get midpoints for samples
    for j in range(num_samples):
        
        n = sample_sizes[j]
        arrays = array_list[j]
        midpoints_temp = np.empty((n, num_pairs, 3))

        for k in range(n):

            vol = arrays[k]
            left_side = vol[lefts]
            right_side = vol[rights]
            mp = np.array([.5*(left_side[j]+right_side[j]) for j in range(num_pairs)])

            midpoints_temp[k] = mp

        midpoints.append(midpoints_temp)
    
    # Get midpoint distances, angles, angle sums for samples
    for j in range(num_samples):
        
        n = sample_sizes[j]
        arrays = array_list[j]
        
        mps = midpoints[j]
     
        angles_temp = np.empty((n, num_pairs-2))
    
        for k in range(n):
            
            vol = arrays[k]
            mp = mps[k]
        
            ANGLES_k = np.empty(0)

            for l in range(1, num_pairs-1):

                # j=1: j-1: head, j: pair 2, j+1: pair 3
                # j=2: j-1: pair 2, j: pair 3, j+1: pair 4
                # ...
                # j=9: j-1: pair 8, j: pair 9, j+1: tail

                A = vol[lefts[l-1]]
                B = vol[rights[l-1]]
                C = vol[lefts[l]]
                D = vol[rights[l]]
                E = vol[lefts[l+1]]
                F = vol[rights[l+1]]

                n_p = mp[l+1]
                c_p = mp[l]
                l_p = mp[l-1]

                n1 = np.cross(D-C, c_p - l_p)
                n1_u = n1 / np.linalg.norm(n1)

                n2 = np.cross(D-C, c_p - n_p)
                n2_u = n2 / np.linalg.norm(n2) 

                r_norm = np.dot(n1_u, n2_u)

                angle = (180/np.pi)*np.arccos(r_norm)

                ANGLES_k = np.append(ANGLES_k, angle)

            angles_temp[k] = ANGLES_k

        angles.append(angles_temp)
    
    return angles

# Statistics

def get_doubles_statistics_QAP(key_list, feature_list, width):
    
    max_key = 1
    num_sets = len(key_list)
    num_bins = int(max_key/width)
    num_features = len(feature_list)
    _, m = feature_list[0][0].shape

    # print('Num 2 pair features:', num_features)

    if num_features > 1:

        start_points = np.arange(0, max_key, width)
        end_points = np.arange(width, max_key+width, width)
        
        cov_matrices = np.empty((num_sets, num_bins, m, num_features, num_features))
        means = np.empty((num_sets, num_bins, m, num_features))
        
        for i in range(num_sets):
            
            this_key_list = key_list[i]
            this_feature_list = [feature_list[z][i] for z in range(num_features)]
                           
            for j in range(num_bins):
                
                start = start_points[j]
                end = end_points[j]

                t1 = np.argwhere(this_key_list >= start)
                t2 = np.argwhere(this_key_list <= end)
                eligible_keys = np.intersect1d(t1, t2)
                
                n_valid = eligible_keys.shape[0]
                # print(n_valid)
                if n_valid - num_features*(num_features-1)> 0:

                    for k in range(m):
                                        
                        eligible_vals = [this_feature_list[i][eligible_keys, k] for i in range(num_features)]
                        darr = np.vstack(tuple(eligible_vals))
                        means[i,j,k] = darr.mean(axis=1)
                        cov_est = np.cov(darr)
                        cov_est_inv = np.linalg.inv(cov_est)
                        cov_matrices[i, j, k] = cov_est_inv
                        # print(darr.shape)
                        # MEANS = darr.mean(axis=1)

                        # if np.isnan(MEANS.sum()):

                        #     means[i,j,k] = 0
                        #     cov_matrices[i, j, k] = 0

                        # else:

                        #     means[i,j,k] = MEANS
                        #     cov_est = np.cov(darr)
                        #     cov_est_inv = np.linalg.inv(cov_est)
                            
                        #     cov_matrices[i, j, k] = cov_est_inv
                else:

                    means[i,j] = 0
                    cov_matrices[i,j] = 0

    elif num_features == 1:

        start_points = np.arange(0, max_key, width)
        end_points = np.arange(width, max_key+width, width)
        
        cov_matrices = np.empty((num_sets, num_bins, m))
        means = np.empty((num_sets, num_bins, m))
        
        for i in range(num_sets):
            
            this_key_list = key_list[i]
            this_feature_list = [feature_list[z][i] for z in range(num_features)]
                           
            for j in range(num_bins):
                
                start = start_points[j]
                end = end_points[j]

                t1 = np.argwhere(this_key_list >= start)
                t2 = np.argwhere(this_key_list <= end)
                eligible_keys = np.intersect1d(t1, t2)

                n_valid = eligible_keys.shape[0]
                # print(n_valid)
                if n_valid > 1:

                    for k in range(m):
                                        
                        eligible_vals = [this_feature_list[i][eligible_keys, k] for i in range(num_features)]
                        darr = np.vstack(tuple(eligible_vals))
                        means[i,j,k] = darr.mean(axis=1)
                        cov_est = np.cov(darr)
                        cov_est_inv = np.var(darr)**-1
                        
                        cov_matrices[i, j, k] = cov_est_inv

                        # MEANS = darr.mean(axis=1)

                        # if np.isnan(MEANS.sum()):

                        #     means[i,j,k] = 0
                        #     cov_matrices[i, j, k] = 0

                        # else:

                        #     means[i,j,k] = MEANS
                        #     cov_est = np.cov(darr)
                        #     cov_est_inv = np.var(darr)**-1
                            
                        #     cov_matrices[i, j, k] = cov_est_inv

                        # # means[i,j,k] = darr.mean(axis=1)

                        # # #cov_est = np.cov(darr)
                        # # #cov_est_inv = np.linalg.inv(cov_est)

                        # # cov_est_inv = np.var(darr) ** -1
                        
                        # # cov_matrices[i, j, k] = cov_est_inv
                else:

                    means[i,j] =0 
                    cov_matrices[i,j] =0 
                
    return means, cov_matrices

def get_doubles_statistics(key_list, feature_list, width):
           
    num_sets = len(key_list)
    num_bins = int(1/width)
    num_features = len(feature_list)
    _, m = feature_list[0][0].shape

    # print('Num', m, 'pair features:', num_features)

    if num_features > 1:

        start_points = np.arange(0, 1, width)
        end_points = np.arange(width, 1+width, width)
        
        cov_matrices = np.empty((num_sets, num_bins, m, num_features, num_features))
        means = np.empty((num_sets, num_bins, m, num_features))
        
        for i in range(num_sets):
            
            this_key_list = key_list[i]
            this_feature_list = [feature_list[z][i] for z in range(num_features)]
                           
            for j in range(num_bins):
                
                start = start_points[j]
                end = end_points[j]

                t1 = np.argwhere(this_key_list >= start)
                t2 = np.argwhere(this_key_list <= end)
                eligible_keys = np.intersect1d(t1, t2)
                
                n_valid = eligible_keys.shape[0]

                if n_valid - num_features*(num_features-1) > 0:

                    for k in range(m):
                                        
                        eligible_vals = [this_feature_list[i][eligible_keys, k] for i in range(num_features)]
                        darr = np.vstack(tuple(eligible_vals))
                        # MEANS = darr.mean(axis=1)
                        means[i,j,k] = darr.mean(axis=1)
                        cov_est = np.cov(darr)
                        cov_est_inv = np.linalg.inv(cov_est)
                        
                        cov_matrices[i, j, k] = cov_est_inv

                        # if np.isnan(MEANS.sum()):

                        #     means[i,j,k] = 0
                        #     cov_matrices[i, j, k] = 0

                        # else:

                        #     means[i,j,k] = MEANS
                        #     cov_est = np.cov(darr)
                        #     cov_est_inv = np.linalg.inv(cov_est)
                            
                        #     cov_matrices[i, j, k] = cov_est_inv

                        # # means[i,j,k] = darr.mean(axis=1)

                        # # cov_est = np.cov(darr)
                        # # cov_est_inv = np.linalg.inv(cov_est)
                        
                        # # cov_matrices[i, j, k] = cov_est_inv
                else:

                    means[i,j] =0 
                    cov_matrices[i,j] = 0

    elif num_features == 1:

        start_points = np.arange(0, 1, width)
        end_points = np.arange(width, 1+width, width)
        
        # print('here')
        cov_matrices = np.empty((num_sets, num_bins, m, num_features, num_features))
        means = np.empty((num_sets, num_bins, m, num_features))
        # print(means.shape, cov_matrices.shape)

        for i in range(num_sets):
            
            this_key_list = key_list[i]
            this_feature_list = [feature_list[z][i] for z in range(num_features)]
                           
            for j in range(num_bins):
                
                start = start_points[j]
                end = end_points[j]

                t1 = np.argwhere(this_key_list >= start)
                t2 = np.argwhere(this_key_list <= end)
                eligible_keys = np.intersect1d(t1, t2)
                
                for k in range(m):
                                    
                    eligible_vals = [this_feature_list[i][eligible_keys, k] for i in range(num_features)]
                    darr = np.vstack(tuple(eligible_vals))
                    means[i,j,k] = darr.mean(axis=1)

                    #cov_est = np.cov(darr)
                    #cov_est_inv = np.linalg.inv(cov_est)

                    cov_est_inv = np.var(darr) ** -1
                    
                    cov_matrices[i, j, k] = cov_est_inv
                
    return means, cov_matrices

def get_time(key, interval):
    
    key = max([key, .001])

    return int(np.floor((key - .00001)/interval))

def get_tl_cost(data, i, t, mu_seq, si_seq, partial_lattice):
    
    seq_mu = mu_seq[i,t]
    seq_si = si_seq[i,t]

    seq_x = get_x_seq_exp(partial_lattice, data)
    seq_cost = get_m_dist(seq_x, seq_mu, seq_si)
    
    return seq_cost

def get_correct_lattice_costs(keys, arrays, width, mu_seq, si_seq, partial_lattice):
    
    tl_costs = []
    
    num_sets = len(keys)
        
    for i in range(num_sets):
        
        these_keys = keys[i]
        these_arrays = arrays[i]
        
        num_obs = len(these_keys)
        corr_costs = np.empty(num_obs)
        
        for j in range(num_obs):
            
            key = these_keys[j]
            data = these_arrays[j]
            
            t = get_time(key, width)

            cost = get_tl_cost(data, i, t, mu_seq, si_seq, partial_lattice)
            
            corr_costs[j] = cost
        
        tl_costs.append(corr_costs)
        
    return tl_costs         

def get_partial_sequence_statistics(key_list, features_doubles, features_triples, width, num_pairs):

    num_sets = len(key_list)
    num_bins = int(1/width)
    num_doubles_features = len(features_doubles)
    num_triples_features = len(features_triples)
    num_features = num_doubles_features + num_triples_features

    start_points = np.arange(0, 1, width)
    end_points = np.arange(width, 1+width, width)
    
    cov_matrices = np.empty((num_sets, num_bins, num_pairs-2, num_features, num_features))
    means = np.empty((num_sets, num_bins, num_pairs-2, num_features))

    for i in range(num_sets):
        
        this_key_list = key_list[i]
        this_feature_list_doubles = [features_doubles[z][i] for z in range(num_doubles_features)]
        this_feature_list_triples = [features_triples[z][i] for z in range(num_triples_features)]
                       
        for j in range(num_bins):
            
            start = start_points[j]
            end = end_points[j]

            t1 = np.argwhere(this_key_list >= start)
            t2 = np.argwhere(this_key_list <= end)
            eligible_keys = np.intersect1d(t1, t2)

            n_valid = eligible_keys.shape[0]

            if n_valid > 0:

                eligible_doubles_vals = [this_feature_list_doubles[i][eligible_keys] for i in range(num_doubles_features)]
                eligible_triples_vals = [this_feature_list_triples[i][eligible_keys] for i in range(num_triples_features)]

                for k in range(num_pairs-2): # Don't need to estimate for last pair. 

                    eligible_doubles_vals_up_to_pair = [a[:,:k+2].sum(axis=1) for a in eligible_doubles_vals]
                    eligible_triples_vals_up_to_pair = [a[:,:k+1].sum(axis=1) for a in eligible_triples_vals]

                    eligible_vals = eligible_doubles_vals_up_to_pair + eligible_triples_vals_up_to_pair
                    darr = np.vstack(tuple(eligible_vals))

                    means[i,j,k] = darr.mean(axis=1)

                    cov_est = np.cov(darr)
                    cov_est_inv = np.linalg.inv(cov_est)

                    cov_matrices[i,j,k] = cov_est_inv

            else:

                means[i,j] = 0
                cov_matrices[i,j] =0 
                
    return means, cov_matrices

def get_sequence_cost_estimate_statistics(key_list, array_list, width, means_partial_sequence, sigma_invs_partial_sequence, tl):
    
    ns = [len(z) for z in key_list]
    n = sum(ns)
    num_pairs = len(tl)
    
    num_datasets = len(key_list)
    
    all_costs = []

    for i in range(num_pairs-2):
        
        partial_lattice = tl[:i+3]
        this_means_partial_sequence = means_partial_sequence[:,:,i,:]
        this_sigma_invs_partial_sequence = sigma_invs_partial_sequence[:,:,i,:]
        
        est_lc = get_correct_lattice_costs(key_list, array_list, width, this_means_partial_sequence, this_sigma_invs_partial_sequence, partial_lattice)
        
        this_i = []
        
        for j in range(num_datasets):
            
            this_n = ns[j]
            all_n_but_this_ds = n - this_n
            
            all_lcs_but_this_ds = [est_lc[z] for z in range(num_datasets) if z!=j]
            all_lcs_but_this_ds_stacked = np.array([c for b in all_lcs_but_this_ds for c in b])
            
            this_i.append(all_lcs_but_this_ds_stacked)
            
        all_costs.append(this_i)
         
    mean_cost_estimates = np.empty((num_datasets, num_pairs-2))
    sigma_inv_cost_estimates = np.empty((num_datasets, num_pairs-2))
    
    for i in range(num_datasets):
        
        cost_estimates = np.array([all_costs[j][i] for j in range(num_pairs-2)])
        cost_estimate_ratios = cost_estimates[-1] / cost_estimates
        log_cost_estimate_ratios = np.log(cost_estimate_ratios)

        mean_cost_estimates_i = log_cost_estimate_ratios.mean(axis=1)
        sigma_cost_estimates_i = log_cost_estimate_ratios.std(axis=1)

        mean_cost_estimates[i] = mean_cost_estimates_i
        sigma_inv_cost_estimates[i] = sigma_cost_estimates_i**-1

    return mean_cost_estimates, sigma_inv_cost_estimates
