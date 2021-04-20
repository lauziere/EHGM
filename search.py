
import numpy as np
import pandas as pd
import os
import time
import sys

from build_arrays import *
from features import *
from util import *
from config import *

np.set_printoptions(suppress=True)

if __name__ == '__main__':

    # Load chosen worm & model information:
    home = config['home']
    dataset = config['dataset']
    seed_nuclei = config['seed_nuclei']
    experiment_name = config['experiment_name']
    model = config['model']
    start_idx = config['start_idx']
    end_idx = config['end_idx']
    time_limit = config['time_limit']
    num_save = config['num_save']
    initial_cost = config['initial_cost']

    resources_path = os.path.join(home, 'Exact_HGM', 'Resources', model)
    data_path = os.path.join(home, 'Exact_HGM', 'Data')
    key_path = os.path.join(data_path, 'keys.npy')
    array_path = os.path.join(data_path, 'arrays.npy')
    
    keys_all = np.load(key_path, allow_pickle=True)
    arrays_all = np.load(array_path, allow_pickle=True)

    keys = keys_all[dataset]
    arrays = arrays_all[dataset]

    max_frames = len(keys)
    end_idx = min(end_idx, max_frames)

    print('\n')
    print('Start Index:', start_idx)
    print('End Index:', end_idx,'\n')

    for i in range(start_idx, end_idx):

        t = keys[i]
        nuclei_arr = arrays[i]

        n = nuclei_arr.shape[0]
        m = int(n/2)

        # Define the correct lattice
        tl = [(i,i+1) for i in range(0,n,2)][::-1]

        # Find the point in embryogenesis:
        interval = get_time(t,.05)

        print('Index:', i)
        print('Key:', np.round(t,2))
        print('Interval:', interval)
        print('n:', n)
        print('Dataset:', dataset)
        print('Model:', model)
        print('Seeded nuclei:', seed_nuclei)
        print('Initial cost:', initial_cost, '\n')

        pairs = get_P_MP(nuclei_arr)
        P = subset_P_seeds(pairs, seed_nuclei, tl)
        P = order_tail_pairs(P, nuclei_arr)

        n_tails = len(P[0])

        best_cost = initial_cost
        best_seqs = np.zeros((num_save, m, 2),'int')
        best_costs = initial_cost*np.ones(num_save)

        if model == 'QAP':

            if n == 20:

                op_means_widths = os.path.join(resources_path, 'means_widths.npy')
                op_sigma_invs_widths = os.path.join(resources_path, 'sigma_invs_widths.npy')
                op_means_pairs = os.path.join(resources_path, 'means_pairs.npy')
                op_sigma_invs_pairs = os.path.join(resources_path, 'sigma_invs_pairs.npy')

            elif n == 22:

                op_means_widths = os.path.join(resources_path, 'means_widths_Q.npy')
                op_sigma_invs_widths = os.path.join(resources_path, 'sigma_invs_widths_Q.npy')
                op_means_pairs = os.path.join(resources_path, 'means_pairs_Q.npy')
                op_sigma_invs_pairs = os.path.join(resources_path, 'sigma_invs_pairs_Q.npy')

            
            means_widths = np.load(op_means_widths)
            sigma_invs_widths = np.load(op_sigma_invs_widths)
            means_pairs = np.load(op_means_pairs)
            sigma_invs_pairs = np.load(op_sigma_invs_pairs)

            # This dataset
            this_means_widths = means_widths[dataset]
            this_sigma_invs_widths = sigma_invs_widths[dataset]
            this_means_pairs = means_pairs[dataset]
            this_sigma_invs_pairs = sigma_invs_pairs[dataset]

            this_interval_means_widths = this_means_widths[interval][0]
            this_interval_sigma_invs_widths = this_sigma_invs_widths[interval][0]
            this_interval_means_pairs = this_means_pairs[interval]
            this_interval_sigma_invs_pairs = this_sigma_invs_pairs[interval]

            H = get_H_QAP(nuclei_arr, P, this_interval_means_pairs, this_interval_sigma_invs_pairs)
            connections = order_H(P,H)
            
            start_time = time.time()

            for j in range(n_tails):

                run_time = time.time() - start_time

                source = P[0][j]

                print('Tail:', j+1, 'of', n_tails, '-', source)
                print('Best Cost:', best_cost)

                HGM = Exact_HGM(nuclei_arr, H, connections, best_cost, source, start_time, time_limit, num_save, best_seqs, best_costs)
                EHGM = QAP(nuclei_arr, H, connections, best_cost, source, start_time, time_limit, num_save, this_interval_means_widths, this_interval_sigma_invs_widths, best_seqs, best_costs)
                EHGM.Begin_Search()
                best_seqs = EHGM.best_seqs
                best_costs = EHGM.best_costs
                best_cost = best_costs[0]

        elif model == 'Pairs':

            if n == 20:

                op_means_pairs = os.path.join(resources_path, 'means_pairs.npy')
                op_sigma_invs_pairs = os.path.join(resources_path, 'sigma_invs_pairs.npy')
                op_means_triples = os.path.join(resources_path, 'means_triples.npy')
                op_sigma_invs_triples = os.path.join(resources_path, 'sigma_invs_triples.npy')

            elif n == 22:

                op_means_pairs = os.path.join(resources_path, 'means_pairs_Q.npy')
                op_sigma_invs_pairs = os.path.join(resources_path, 'sigma_invs_pairs_Q.npy')
                op_means_triples = os.path.join(resources_path, 'means_triples_Q.npy')
                op_sigma_invs_triples = os.path.join(resources_path, 'sigma_invs_triples_Q.npy')
            
            means_pairs = np.load(op_means_pairs)
            sigma_invs_pairs = np.load(op_sigma_invs_pairs)
            means_triples = np.load(op_means_triples)
            sigma_invs_triples = np.load(op_sigma_invs_triples)

            # This dataset
            this_means_pairs = means_pairs[dataset]
            this_sigma_invs_pairs = sigma_invs_pairs[dataset]
            this_means_triples = means_triples[dataset]
            this_sigma_invs_triples = sigma_invs_triples[dataset]

            this_interval_means_pairs = this_means_pairs[interval]
            this_interval_sigma_invs_pairs = this_sigma_invs_pairs[interval]
            this_interval_means_triples = this_means_triples[interval]
            this_interval_sigma_invs_triples = this_sigma_invs_triples[interval]

            H = get_H_Pairs(nuclei_arr, P, this_interval_means_pairs, this_interval_sigma_invs_pairs)
            connections = order_H(P,H)

            start_time = time.time()

            for j in range(n_tails):

                run_time = time.time() - start_time

                source = P[0][j]

                print('Tail:', j+1, 'of', n_tails, '-', source)
                print('Best Cost:', best_cost)

                HGM = Exact_HGM(nuclei_arr, H, connections, best_cost, source, start_time, time_limit, num_save, best_seqs, best_costs)
                EHGM = Pairs(nuclei_arr, H, connections, best_cost, source, start_time, time_limit, num_save, this_interval_means_triples, this_interval_sigma_invs_triples, best_seqs, best_costs)
                EHGM.Begin_Search()
                best_seqs = EHGM.best_seqs
                best_costs = EHGM.best_costs
                best_cost = best_costs[0]

        elif model == 'Full':

            resources_path = os.path.join(home, 'Exact_HGM', 'Resources')

            if n == 20:

                ip_means_pairs = os.path.join(resources_path, 'Pairs', 'means_pairs.npy')
                ip_sigma_invs_pairs = os.path.join(resources_path, 'Pairs', 'sigma_invs_pairs.npy')
                ip_means_seq = os.path.join(resources_path, 'Full', 'means_sequence.npy')
                ip_sigma_invs_seq = os.path.join(resources_path, 'Full', 'sigma_invs_sequence.npy')

                ip_mean_cost_estimates = os.path.join(resources_path, 'Full', 'mean_cost_estimates.npy')
                ip_sigma_inv_cost_estimates = os.path.join(resources_path, 'Full', 'sigma_inv_cost_estimates.npy')

            elif n == 22:

                ip_means_pairs = os.path.join(resources_path, 'Pairs', 'means_pairs_Q.npy')
                ip_sigma_invs_pairs = os.path.join(resources_path, 'Pairs', 'sigma_invs_pairs_Q.npy')
                ip_means_seq = os.path.join(resources_path, 'Full', 'means_sequence_Q.npy')
                ip_sigma_invs_seq = os.path.join(resources_path, 'Full', 'sigma_invs_sequence_Q.npy')

                ip_mean_cost_estimates = os.path.join(resources_path, 'Full', 'mean_cost_estimates_Q.npy')
                ip_sigma_inv_cost_estimates = os.path.join(resources_path, 'Full', 'sigma_inv_cost_estimates_Q.npy')
            
            means_pairs = np.load(ip_means_pairs)
            sigma_invs_pairs = np.load(ip_sigma_invs_pairs)
            means_sequence = np.load(ip_means_seq)
            sigma_invs_sequence = np.load(ip_sigma_invs_seq)

            mean_cost_estimates = np.load(ip_mean_cost_estimates)
            sigma_inv_cost_estimates = np.load(ip_sigma_inv_cost_estimates)

            this_means_pairs = means_pairs[dataset]
            this_sigma_invs_pairs = sigma_invs_pairs[dataset]
            this_means_sequence = means_sequence[dataset]
            this_sigma_invs_sequence = sigma_invs_sequence[dataset]

            this_mean_cost_est = mean_cost_estimates[dataset]
            this_sigma_inv_cost_est = sigma_inv_cost_estimates[dataset]

            this_interval_means_pairs = this_means_pairs[interval] 
            this_interval_sigma_invs_pairs = this_sigma_invs_pairs[interval] 
            this_interval_means_sequence = this_means_sequence[interval] 
            this_interval_sigma_invs_sequence = this_sigma_invs_sequence[interval]

            H = get_H_Pairs(nuclei_arr, P, this_interval_means_pairs, this_interval_sigma_invs_pairs)
            connections = order_H(P,H)

            start_time = time.time()

            for j in range(n_tails):

                run_time = time.time() - start_time

                source = P[0][j]

                print('Tail:', j+1, 'of', n_tails, '-', source)
                print('Best Cost:', best_cost)

                HGM = Exact_HGM(nuclei_arr, H, connections, best_cost, source, start_time, time_limit, num_save, best_seqs, best_costs)
                EHGM = Full(nuclei_arr, H, connections, best_cost, source, start_time, time_limit, num_save, this_interval_means_sequence, this_interval_sigma_invs_sequence, this_mean_cost_est, this_sigma_inv_cost_est, best_seqs, best_costs)
                EHGM.Begin_Search()
                best_seqs = EHGM.best_seqs
                best_costs = EHGM.best_costs
                best_cost = best_costs[0]

        elif model == 'PF':

            resources_path = os.path.join(home, 'Exact_HGM', 'Resources')

            if n == 20:

                ip_means_pairs = os.path.join(resources_path, 'Pairs', 'means_pairs.npy')
                ip_sigma_invs_pairs = os.path.join(resources_path, 'Pairs', 'sigma_invs_pairs.npy')
                ip_means_triples = os.path.join(resources_path, 'Pairs', 'means_triples.npy')
                ip_sigma_invs_triples = os.path.join(resources_path, 'Pairs', 'sigma_invs_triples.npy')
                ip_means_seq = os.path.join(resources_path, 'Full', 'means_sequence.npy')
                ip_sigma_invs_seq = os.path.join(resources_path, 'Full', 'sigma_invs_sequence.npy')

            elif n == 22:

                ip_means_pairs = os.path.join(resources_path, 'Pairs', 'means_pairs_Q.npy')
                ip_sigma_invs_pairs = os.path.join(resources_path, 'Pairs', 'sigma_invs_pairs_Q.npy')
                ip_means_triples = os.path.join(resources_path, 'Pairs', 'means_triples_Q.npy')
                ip_sigma_invs_triples = os.path.join(resources_path, 'Pairs', 'sigma_invs_triples_Q.npy')
                ip_means_seq = os.path.join(resources_path, 'Full', 'means_sequence_Q.npy')
                ip_sigma_invs_seq = os.path.join(resources_path, 'Full', 'sigma_invs_sequence_Q.npy')
            
            means_pairs = np.load(ip_means_pairs)
            sigma_invs_pairs = np.load(ip_sigma_invs_pairs)
            means_triples = np.load(ip_means_triples)
            sigma_invs_triples = np.load(ip_sigma_invs_triples)
            means_sequence = np.load(ip_means_seq)
            sigma_invs_sequence = np.load(ip_sigma_invs_seq)

            this_means_pairs = means_pairs[dataset]
            this_sigma_invs_pairs = sigma_invs_pairs[dataset]
            this_means_triples = means_triples[dataset]
            this_sigma_invs_triples = sigma_invs_triples[dataset]
            this_means_sequence = means_sequence[dataset]
            this_sigma_invs_sequence = sigma_invs_sequence[dataset]

            this_interval_means_pairs = this_means_pairs[interval] 
            this_interval_sigma_invs_pairs = this_sigma_invs_pairs[interval] 
            this_interval_means_triples = this_means_triples[interval]
            this_interval_sigma_invs_triples = this_sigma_invs_triples[interval]
            this_interval_means_sequence = this_means_sequence[interval] 
            this_interval_sigma_invs_sequence = this_sigma_invs_sequence[interval]

            this_interval_means_sequence_full = this_interval_means_sequence[-1]
            this_interval_sigma_invs_sequence_full = this_interval_sigma_invs_sequence[-1]

            H = get_H_Pairs(nuclei_arr, P, this_interval_means_pairs, this_interval_sigma_invs_pairs)
            connections = order_H(P,H)

            start_time = time.time()

            for j in range(n_tails):

                run_time = time.time() - start_time

                source = P[0][j]

                print('Tail:', j+1, 'of', n_tails, '-', source)
                print('Best Cost:', best_cost)

                HGM = Exact_HGM(nuclei_arr, H, connections, best_cost, source, start_time, time_limit, num_save, best_seqs, best_costs)
                EHGM = PF(nuclei_arr, H, connections, best_cost, source, start_time, time_limit, num_save, this_interval_means_triples, this_interval_sigma_invs_triples, this_interval_means_sequence_full, this_interval_sigma_invs_sequence_full, best_seqs, best_costs)
                EHGM.Begin_Search()
                best_seqs = EHGM.best_seqs
                best_costs = EHGM.best_costs
                best_cost = best_costs[0]
        
        # Store Results:
        runtime = EHGM.elapsed_time

        print('\n')
        print('Runtime:', np.round(runtime,2))
        for j in range(num_save):
            print(j+1, best_seqs[j].T, best_costs[j], '\n')

        out_path = os.path.join(home, 'Exact_HGM', 'Results', experiment_name, model, str(dataset), str(i))

        if not os.path.isdir(out_path):

            os.makedirs(out_path)

        out_path_seqs = os.path.join(out_path, 'best_sequences.npy')
        out_path_costs = os.path.join(out_path, 'best_costs.npy')
        out_path_runtime = os.path.join(out_path, 'runtime.npy')

        np.save(out_path_seqs, best_seqs)
        np.save(out_path_costs, best_costs)
        np.save(out_path_runtime, runtime)





