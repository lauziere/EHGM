
import numpy as np
import pandas as pd
import os

from features import *
from util import *

# Handle datasets
def merge_data(key_list, array_list):
    
    all_keys = np.array([kl[i] for kl in key_list for i in range(len(kl))])
    all_keys_idxs = np.argsort(all_keys)
    all_keys_sorted = np.sort(all_keys)
    
    all_arrays = np.array([al[i] for al in array_list for i in range(len(al))])
    all_arrays_shape = all_arrays.shape
    
    n = len(all_arrays)
    sorted_arrays = np.empty(all_arrays_shape)
    
    for j in all_keys_idxs:
        
        this_idx = all_keys_idxs[j]
        sorted_arrays[j] = all_arrays[this_idx]
        
    return all_keys_sorted, sorted_arrays

def merge_data_LOO(key_list, array_list):
    
    num_sets = len(key_list)
    
    key_list_LOO = []
    array_list_LOO = []
    
    for i in range(num_sets):
        
        remaining_keys = [key_list[j] for j in range(num_sets) if j!=i]
        remaining_arrays = [array_list[j] for j in range(num_sets) if j!=i]
        
        a,b = merge_data(remaining_keys, remaining_arrays)
        
        key_list_LOO.append(a)
        array_list_LOO.append(b)
        
    return key_list_LOO, array_list_LOO

if __name__ == '__main__':

    # base_dir = os.path.abspath('.')
    base_dir = '/data/lauzierean/Untwisting'
    home = os.path.join(base_dir, 'exacthgm')
    data_dir = os.path.join(home, 'Data')

    resources = os.path.join(home, 'Resources')
    resources_QAP = os.path.join(resources,'QAP')
    resources_Pairs = os.path.join(resources,'Pairs')
    resources_Full = os.path.join(resources,'Full')

    # twitches = [5, 8, 23] + [11, 52, 50] + [15, 12, 11] + [37] + [9, 15, 13] + [1, 4, 1]
    # hatches = [85, 106, 136] + [102, 138, 140] + [104, 101, 126] + [114] + [93, 96, 93] + [86, 95, 95]

    # twitches = np.array(twitches)
    # hatches = np.array(hatches)

    # twitch_hatch = np.vstack([twitches, hatches])
    # out_path_twitch_hatch = os.path.join(resources, 'twitch_hatch.npy')
    # np.save(out_path_twitch_hatch, twitch_hatch)

    step = .05

    # First do the n=20 nuclei case:
    num_cells = 20
    num_pairs = int(num_cells/2)
    tl = [(i,i+1) for i in range(0,num_cells,2)][::-1]

    key_list = np.load(os.path.join(data_dir, 'keys_20.npy'), allow_pickle=True)
    array_list = np.load(os.path.join(data_dir, 'arrays_20.npy'), allow_pickle=True)

    # Get LOO arrays to get statistics
    keys_LOO, arrays_LOO = merge_data_LOO(key_list, array_list)

    # QAP
    widths = get_pair_arrays(keys_LOO, arrays_LOO)
    widths_pairs = [a[:,1:] for a in widths] 

    side_lens = get_side_lens(keys_LOO, arrays_LOO)
    left_sides, right_sides = side_lens

    feature_list_widths = [widths]
    means_widths, sigma_invs_widths = get_doubles_statistics_QAP(keys_LOO, feature_list_widths, step)

    feature_list_pairs = [widths_pairs, left_sides, right_sides]
    means_pairs, sigma_invs_pairs = get_doubles_statistics_QAP(keys_LOO, feature_list_pairs, step)

    op_means_widths = os.path.join(resources_QAP, 'means_widths.npy')
    op_sigma_invs_widths = os.path.join(resources_QAP, 'sigma_invs_widths.npy')
    op_means_pairs = os.path.join(resources_QAP, 'means_pairs.npy')
    op_sigma_invs_pairs = os.path.join(resources_QAP, 'sigma_invs_pairs.npy')

    np.save(op_means_widths, means_widths)
    np.save(op_sigma_invs_widths, sigma_invs_widths)
    np.save(op_means_pairs, means_pairs)
    np.save(op_sigma_invs_pairs, sigma_invs_pairs)

    # Pairs
    # Degree 4 features:
    pair_dist_ratios = get_pair_ratio_arrays(keys_LOO, arrays_LOO)
    pair_dist_ratios = [a[:,1:] for a in pair_dist_ratios] 

    cos_sims =  get_cos_sims(keys_LOO, arrays_LOO)
    lateral_axial_twists = get_lateral_axial_twists(keys_LOO, arrays_LOO)
    axial_twists = get_axial_twists(keys_LOO, arrays_LOO)

    feature_list_pairs = [cos_sims, lateral_axial_twists, axial_twists]
    means_pairs, sigma_invs_pairs = get_doubles_statistics(keys_LOO, feature_list_pairs, step)

    # Degree 6 features:
    midpoints, midpoint_distances, bend_angles, bend_angle_sums = get_bend_angles(keys_LOO, arrays_LOO)
    midpoint_distances = [a[:,1:] for a in midpoint_distances]
    plane_angles = get_plane_intersection_angles(keys_LOO, arrays_LOO)

    feature_list_triples = [pair_dist_ratios, midpoint_distances, bend_angles, plane_angles]
    means_triples, sigma_invs_triples = get_doubles_statistics(keys_LOO, feature_list_triples, step)

    op_means_pairs = os.path.join(resources_Pairs, 'means_pairs.npy')
    op_sigma_invs_pairs = os.path.join(resources_Pairs, 'sigma_invs_pairs.npy')
    op_means_triples = os.path.join(resources_Pairs, 'means_triples.npy')
    op_sigma_invs_triples = os.path.join(resources_Pairs, 'sigma_invs_triples.npy')

    np.save(op_means_pairs, means_pairs)
    np.save(op_sigma_invs_pairs, sigma_invs_pairs)
    np.save(op_means_triples, means_triples)
    np.save(op_sigma_invs_triples, sigma_invs_triples)

    # Full
    means_partial_sequence, sigma_invs_partial_sequence = get_partial_sequence_statistics(keys_LOO, feature_list_pairs, feature_list_triples, step, num_pairs)
    mean_cost_estimates, sigma_inv_cost_estimates = get_sequence_cost_estimate_statistics(key_list, array_list, step, means_partial_sequence, sigma_invs_partial_sequence, tl)

    op_means_seq = os.path.join(resources_Full, 'means_sequence.npy')
    op_sigma_invs_seq = os.path.join(resources_Full, 'sigma_invs_sequence.npy')
    op_mean_cost_estimates = os.path.join(resources_Full, 'mean_cost_estimates.npy')
    op_sigma_cost_estimates = os.path.join(resources_Full, 'sigma_inv_cost_estimates.npy')

    np.save(op_means_seq, means_partial_sequence)
    np.save(op_sigma_invs_seq, sigma_invs_partial_sequence)
    np.save(op_mean_cost_estimates, mean_cost_estimates)
    np.save(op_sigma_cost_estimates, sigma_inv_cost_estimates)

    # Now 22 cell case:
    num_cells = 22
    num_pairs = int(num_cells/2)
    tl = [(i,i+1) for i in range(0,num_cells,2)][::-1]

    key_list = np.load(os.path.join(data_dir, 'keys_22.npy'), allow_pickle=True)
    array_list = np.load(os.path.join(data_dir, 'arrays_22.npy'), allow_pickle=True)

    # Get LOO arrays to get statistics
    keys_LOO, arrays_LOO = merge_data_LOO(key_list, array_list)

    # QAP
    widths = get_pair_arrays(keys_LOO, arrays_LOO)
    widths_pairs = [a[:,1:] for a in widths] 

    side_lens = get_side_lens(keys_LOO, arrays_LOO)
    left_sides, right_sides = side_lens

    feature_list_widths = [widths]
    means_widths, sigma_invs_widths = get_doubles_statistics_QAP(keys_LOO, feature_list_widths, step)

    feature_list_pairs = [widths_pairs, left_sides, right_sides]
    means_pairs, sigma_invs_pairs = get_doubles_statistics_QAP(keys_LOO, feature_list_pairs, step)

    op_means_widths = os.path.join(resources_QAP, 'means_widths_Q.npy')
    op_sigma_invs_widths = os.path.join(resources_QAP, 'sigma_invs_widths_Q.npy')
    op_means_pairs = os.path.join(resources_QAP, 'means_pairs_Q.npy')
    op_sigma_invs_pairs = os.path.join(resources_QAP, 'sigma_invs_pairs_Q.npy')

    np.save(op_means_widths, means_widths)
    np.save(op_sigma_invs_widths, sigma_invs_widths)
    np.save(op_means_pairs, means_pairs)
    np.save(op_sigma_invs_pairs, sigma_invs_pairs)

    # Pairs
    # Degree 4 features:
    pair_dist_ratios = get_pair_ratio_arrays(keys_LOO, arrays_LOO)
    pair_dist_ratios = [a[:,1:] for a in pair_dist_ratios] 

    cos_sims =  get_cos_sims(keys_LOO, arrays_LOO)
    lateral_axial_twists = get_lateral_axial_twists(keys_LOO, arrays_LOO)
    axial_twists = get_axial_twists(keys_LOO, arrays_LOO)

    feature_list_pairs = [cos_sims, lateral_axial_twists, axial_twists]
    means_pairs, sigma_invs_pairs = get_doubles_statistics(keys_LOO, feature_list_pairs, step)

    # Degree 6 features:
    midpoints, midpoint_distances, bend_angles, bend_angle_sums = get_bend_angles(keys_LOO, arrays_LOO)
    midpoint_distances = [a[:,1:] for a in midpoint_distances]
    plane_angles = get_plane_intersection_angles(keys_LOO, arrays_LOO)

    feature_list_triples = [pair_dist_ratios, midpoint_distances, bend_angles, plane_angles]
    means_triples, sigma_invs_triples = get_doubles_statistics(keys_LOO, feature_list_triples, step)

    op_means_pairs = os.path.join(resources_Pairs, 'means_pairs_Q.npy')
    op_sigma_invs_pairs = os.path.join(resources_Pairs, 'sigma_invs_pairs_Q.npy')
    op_means_triples = os.path.join(resources_Pairs, 'means_triples_Q.npy')
    op_sigma_invs_triples = os.path.join(resources_Pairs, 'sigma_invs_triples_Q.npy')

    np.save(op_means_pairs, means_pairs)
    np.save(op_sigma_invs_pairs, sigma_invs_pairs)
    np.save(op_means_triples, means_triples)
    np.save(op_sigma_invs_triples, sigma_invs_triples)

    # Full
    means_partial_sequence, sigma_invs_partial_sequence = get_partial_sequence_statistics(keys_LOO, feature_list_pairs, feature_list_triples, step, num_pairs)
    mean_cost_estimates, sigma_inv_cost_estimates = get_sequence_cost_estimate_statistics(key_list, array_list, step, means_partial_sequence, sigma_invs_partial_sequence, tl)

    op_means_seq = os.path.join(resources_Full, 'means_sequence_Q.npy')
    op_sigma_invs_seq = os.path.join(resources_Full, 'sigma_invs_sequence_Q.npy')
    op_mean_cost_estimates = os.path.join(resources_Full, 'mean_cost_estimates_Q.npy')
    op_sigma_cost_estimates = os.path.join(resources_Full, 'sigma_inv_cost_estimates_Q.npy')

    np.save(op_means_seq, means_partial_sequence)
    np.save(op_sigma_invs_seq, sigma_invs_partial_sequence)
    np.save(op_mean_cost_estimates, mean_cost_estimates)
    np.save(op_sigma_cost_estimates, sigma_inv_cost_estimates)