
import numpy as np
import time
from scipy.spatial import distance

# from data_handling import *
from features import *
from config import *

# Distance matrix
def get_nuclei_distance_matrix(nuclei_arr):

    n = nuclei_arr.shape[0] # 20 or 22
    distance_matrix = np.zeros((n, n))

    for i in range(n):

        nucleus_i = nuclei_arr[i]

        for j in range(n):

            nucleus_j = nuclei_arr[j]

            if i < j:

                distance_matrix[i,j] = np.linalg.norm(nucleus_i - nucleus_j)
                distance_matrix[j,i] = distance_matrix[i,j]

    return distance_matrix

# Get pairs
def get_P(n):

    pairs = [(i,j) for i in range(n) for j in range(n) if i!=j]

    return pairs

def midpoint_test(nuclei_arr, pair):

    '''
    The midpoint test calculates the midpoint between the pair of nuceli. If there exists another nuclei centroid
    that is closer to the midpoint than the potential pair, the pair is ruled out. If it passes the test for all other
    centroids then it is kept as a potential pair.
    '''

    n = nuclei_arr.shape[0]
    left, right = pair
    left_point = nuclei_arr[left]
    right_point = nuclei_arr[right]

    midpoint = .5*(left_point + right_point)
    midpoint_dist = np.linalg.norm(left_point - midpoint)

    remaining_points = np.setdiff1d(np.arange(n), [left, right])
    nuclei_arr_sub = nuclei_arr[remaining_points]

    for i in range(n-2):

        this_point = nuclei_arr[i]
        check_dist = np.linalg.norm(this_point-midpoint)

        if check_dist < .5*midpoint_dist:

            return False

    return True

def get_P_MP(nuclei_arr):

    n = nuclei_arr.shape[0]

    pairs = [(i,j) for i in range(n) for j in range(n) if (i!=j) & (midpoint_test(nuclei_arr, (i,j)))]

    return pairs

def subset_P_seeds(pairs, seed_nuclei, tl):

    m = len(tl)
    n = 2*m

    tl_flatten = [a for b in tl for a in b]

    # Initialize P
    P = [pairs for i in range(m)]

    seed_nuclei = np.array(seed_nuclei)
    seed_nuclei = seed_nuclei[(seed_nuclei < n) & (seed_nuclei >= 0)]

    for s in seed_nuclei:

        side = 1 if s%2 else 0 # left or right 
        lat_pt = tl_flatten.index(s)
        pair = int(lat_pt/2)

        for i in range(m):

            this_vertex_set = P[i]

            if i != pair:

                subset_vertex_set = [a for a in this_vertex_set if (a[0]!=s) and (a[1]!=s)]
                P[i] = subset_vertex_set

            elif i == pair:

                subset_vertex_set = [a for a in this_vertex_set if a[side]==s]
                P[i] = subset_vertex_set

    return P


    # If given a seed, say 0, then:
    # any pair other than the last with 0 is removed AND and pair in the last without 0 in the correct position is removed. 
    # - link 0 to pair m

def order_tail_pairs(P, nuclei_arr):

    dist_mat = get_nuclei_distance_matrix(nuclei_arr)

    tail_set = P[0]

    tail_dists = [dist_mat[a[0],a[1]] for a in tail_set]
    ordered_tail_dists = np.argsort(tail_dists)

    ordered_tails = [tail_set[z] for z in ordered_tail_dists]
    P[0] = ordered_tails

    return P

def get_m_dist(x, mu, sigma_inv):

    a = (x - mu).T
    b = (x - mu)
    
    g = np.linalg.multi_dot([a, sigma_inv, b])

    return g

def order_H(P, H):

    m = len(P)
    edge_dicts = []

    for i in range(m-1):

        this_H = H[i] 
        this_H_pairs = list(this_H.keys())
        this_P = P[i] 

        edge_dict = {}

        for pair in this_P: 

            out_connects = [a[1] for a in this_H_pairs if a[0] == pair] 
            num_connects = len(out_connects)

            trim_ocs_cos_dists = [this_H[(pair, out_connects[z])] for z in range(num_connects)]
            sorted_trim_ocs = np.argsort(trim_ocs_cos_dists)

            edge_dict[pair] = [out_connects[idx] for idx in sorted_trim_ocs]

        edge_dicts.append(edge_dict)

    return edge_dicts

class Exact_HGM:

    def __init__(self, data, H, connections, best_cost, source, start_time, time_limit, num_save, best_seqs, best_costs):

        # Store args
        self.data = data
        self.n = data.shape[0]
        self.m = int(self.n/2)
        self.H = H
        self.connections = connections
        self.cost_star = best_cost
        self.source = source

        # Create search structures
        self.seq = []
        self.seq_length = 0
        self.seq_star = np.zeros((self.m, 2), 'int')
        self.H_cost = np.empty((self.m-1))
        self.source_cost = 0
        self.cost = self.source_cost

        self.P = [[] for i in range(self.m)]
        self.Q = [[] for i in range(self.m)]
        self.n_P = np.zeros(self.m, 'int')
        self.n_Q = np.zeros(self.m, 'int')

        # start time
        self.start_time = start_time
        self.elapsed_time = time.time() - start_time
        self.time_limit = time_limit

        # Store top lattices
        self.num_save = num_save
        self.best_seqs = best_seqs
        self.best_costs = best_costs

    def lattice_storage(self):

        best_costs = self.best_costs
        best_seqs = self.best_seqs

        costs_merge = np.append(best_costs, self.cost)
        seqs_merge = np.concatenate([best_seqs, np.array(self.seq).reshape((1,self.m,2))],axis=0)
        
        top_idxs = np.argsort(costs_merge)
        top_seqs = seqs_merge[top_idxs][:self.num_save]
        top_costs = costs_merge[top_idxs][:self.num_save]

        self.best_seqs = top_seqs
        self.best_costs = top_costs

    def seq_append(self, vert):

        self.seq.append(vert)
        self.seq_length += 1

    def seq_remove(self):

        self.seq.pop(-1)
        self.seq_length += -1

    def get_top(self, i):

        return self.Q[i][0]

    def Prune(self, i):

        # First prune for assignment constraints:
        P_next = self.P[i].copy()
        seq = self.seq

        used_detections = [a for b in seq for a in b]
        Q_next = [a for a in P_next if a[0] not in used_detections and a[1] not in used_detections]

        # Then prune for cost constraints:
        cost_star = self.cost_star
        cost = self.cost
        cost_dict = self.H[i]

        last = seq[-1]
        Q_next = [a for a in Q_next if cost + cost_dict[(last, a)] <= cost_star]

        # Lattice self intersection test
        # if i >= 1:

        #     Q_next = [a for a in Q_next if self.lattice_intersection(a)]

        self.Q[i] = Q_next
        self.n_Q[i] = len(Q_next)

    def lattice_intersection(self, next_vert):

        partial_m = len(self.seq)

        last_left, last_right = self.seq[-1]
        next_left, next_right = next_vert

        left_points = np.array([self.data[last_left], self.data[next_left]])
        right_points = np.array([self.data[last_right], self.data[next_right]])

        left_mp = left_points.mean()
        right_mp = right_points.mean()

        last_mp = .5*(last_left + last_right)

        intercell_mp = .5*(left_mp + right_mp)

        width = np.linalg.norm(intercell_mp - left_mp)
        length = np.linalg.norm(intercell_mp - last_mp)

        rad = np.min([width, length])

        pass_fail = True

        for j in range(partial_m-1):

            left_start, right_start = self.seq[j]
            left_end, right_end = self.seq[j+1]

            left_points_j = np.array([self.data[left_start], self.data[left_end]])
            right_points_j = np.array([self.data[right_start], self.data[right_end]])

            left_mp_j = left_points_j.mean()
            right_mp_j = right_points_j.mean()

            last_mp_j = .5*(left_start + right_start)
            intercell_mp_j = .5*(left_mp_j + right_mp_j)

            width_j = np.linalg.norm(intercell_mp_j - left_mp_j)
            length_j = np.linalg.norm(intercell_mp_j - last_mp_j)

            rad_j = np.min([width_j, length_j])

            rad_dist = np.linalg.norm(intercell_mp_j - intercell_mp)

            new_mp_too_close_to_old_mp = .1*rad > rad_dist

            if new_mp_too_close_to_old_mp:

                print(.1*rad, rad_dist)
                pass_fail = False
                return pass_fail

        return pass_fail

    def Enqueue(self, i):

        last_pair = self.seq[-1]
        edge_dict = self.connections[i]
        possible_connections = edge_dict[last_pair].copy()

        self.P[i] = possible_connections
        self.n_P[i] = len(self.P[i])
        
        self.Prune(i)

    def clear_queue(self, i):

        self.Q[i].clear()
        self.n_Q[i] = 0

    def remove_specific(self, i, vert):

        self.Q[i].remove(vert)
        self.n_Q[i] += -1

# QAP

def get_x_QAP(lattice, data):
    
    # Load points
    last_left, last_right = lattice[-2]
    this_left, this_right = lattice[-1]
    
    last_left_pt = data[last_left]
    last_right_pt = data[last_right]
    this_left_pt = data[this_left]
    this_right_pt = data[this_right]
    
    # pair distance ratio
    width = np.linalg.norm(this_right_pt - this_left_pt)

    # left length
    left_len = np.linalg.norm(last_left_pt - this_left_pt)

    # right length
    right_len = np.linalg.norm(last_right_pt - this_right_pt)
            
    x = np.array([width, left_len, right_len])
    
    return x 

def get_H_QAP(data, body_pair_sets, mean_vec_doubles, sigma_inv_doubles):

    cost_dicts = []

    m = len(body_pair_sets)

    for i in range(m-1):

        cost_dict = {}

        current_pairs = body_pair_sets[i]
        num_current_pairs = len(current_pairs)

        next_pairs = body_pair_sets[i+1]
        num_next_pairs = len(next_pairs)

        mean_vec = mean_vec_doubles[i]
        sigma_inv = sigma_inv_doubles[i]

        for j in range(num_current_pairs):

            current_pair = current_pairs[j]
            cp_l, cp_r = current_pair

            for k in range(num_next_pairs):

                next_pair = next_pairs[k]
                np_l, np_r = next_pair

                lattice = (current_pair, next_pair)
                cells = [a for b in lattice for a in b]

                t1 = len(set(cells)) == 4

                if t1:

                    x = get_x_QAP(lattice, data)
                    cost_dict[lattice] = get_m_dist(x, mean_vec, sigma_inv)

        cost_dicts.append(cost_dict)

    return cost_dicts

class QAP(Exact_HGM):

    def __init__(self, data, H, connections, best_cost, source, start_time, time_limit, num_save, source_mean, source_sigma_inv, best_seqs, best_costs):

        self.source_mean = source_mean
        self.source_sigma_inv = source_sigma_inv
        self.source_cost = 0

        Exact_HGM.__init__(self, data, H, connections, best_cost, source, start_time, time_limit, num_save, best_seqs, best_costs)

    def get_source_cost(self):

        source = self.source
        left, right = source
        chord_length = np.linalg.norm(self.data[right] - self.data[left])

        return self.source_sigma_inv * (chord_length - self.source_mean)**2 

    def Branch(self, i):
        
        self.elapsed_time = time.time() - self.start_time
        this_vert = self.seq[i]
        self.Enqueue(i)

        while self.n_Q[i] > 0 and self.elapsed_time <= self.time_limit:

            next_vert = self.get_top(i)
            self.seq_append(next_vert)
            this_H_cost = self.H[i][self.seq[-2], self.seq[-1]]
            self.H_cost[i] = this_H_cost
            self.cost += this_H_cost

            i+=1

            if i < self.m - 1:

                self.Branch(i)

            if i == self.m - 1:

                self.cost = self.source_cost + self.H_cost.sum() 

                if self.cost <= self.cost_star:

                    self.seq_star = self.seq.copy()
                    self.cost_star = self.cost

                    print(self.seq_star, np.round(self.cost_star,2),'- Runtime:', np.round(self.elapsed_time,2), 'seconds')

                if (self.cost <= self.best_costs[-1]):

                    self.lattice_storage()


            self.remove_specific(i-1, next_vert)
            self.seq_remove()
            self.H_cost[i-1] = 0
            
            self.cost = self.source_cost + self.H_cost.sum() 

            i += -1

    def Begin_Search(self):

        i = 0
    
        self.seq_append(self.source)
        self.source_cost = self.get_source_cost()
        self.cost = self.source_cost

        self.Branch(i)

# Pairs

def get_m_dist_Pairs(x, mu, sigma_inv):   

    x[0] = np.min([x[0], mu[0]])

    a = (x - mu).T
    b = (x - mu)
    
    g = np.linalg.multi_dot([a, sigma_inv, b])

    return g

def get_x_degree_6(lattice, data):

    j = -2

    last_pair = lattice[j-1]
    last_left, last_right = last_pair
    ll_p = data[last_left]
    lr_p = data[last_right]
    lp_mp = .5*(ll_p + lr_p)

    this_pair = lattice[j]
    this_left, this_right = this_pair
    tl_p = data[this_left]
    tr_p = data[this_right]
    tp_mp = .5*(tl_p + tr_p)

    next_pair = lattice[j+1]
    next_left, next_right = next_pair
    nl_p = data[next_left]
    nr_p = data[next_right]
    np_mp = .5*(nl_p + nr_p)

    pd_ratio = (np.linalg.norm(nr_p - nl_p) / np.linalg.norm(tr_p - tl_p))
    lat_length = np.linalg.norm(np_mp - tp_mp)

    r1 = lp_mp - tp_mp
    r1_mag = np.linalg.norm(r1)

    r2 = np_mp - tp_mp
    r2_mag = np.linalg.norm(r2)

    new_ray = np.dot(r1, r2)
    new_ray_norm = new_ray / (r1_mag * r2_mag)
    new_ang = (180/np.pi)*np.arccos(new_ray_norm)      

    # plane angle sum

    n1 = np.cross(tr_p - tl_p, tp_mp - lp_mp)
    n1_u = n1 / np.linalg.norm(n1)

    n2 = np.cross(tr_p - tl_p, tp_mp - np_mp)
    n2_u = n2 / np.linalg.norm(n2)

    r_norm = np.dot(n1_u, n2_u)
    angle = (180/np.pi)*np.arccos(r_norm)

    plane_angle = angle

    x = np.array([pd_ratio, lat_length, new_ang, plane_angle])

    return x

def get_x_degree_4(lattice, data):
    
    # Load points
    last_left, last_right = lattice[-2]
    this_left, this_right = lattice[-1]
    
    last_left_pt = data[last_left]
    last_right_pt = data[last_right]
    this_left_pt = data[this_left]
    this_right_pt = data[this_right]
    
    # cs
    left_ray = this_left_pt - last_left_pt
    right_ray = this_right_pt - last_right_pt

    left_ray_mag = np.linalg.norm(left_ray)
    right_ray_mag = np.linalg.norm(right_ray)

    ray_dot = np.dot(left_ray, right_ray)
    cs = ray_dot/(left_ray_mag*right_ray_mag)
                
    # andrew lat ax  
    b1 = this_left_pt - last_left_pt 
    b1_u = b1 / np.linalg.norm(b1)

    b2 = last_left_pt - last_right_pt
    b2_u = b2 / np.linalg.norm(b2)

    b3 = last_right_pt - this_right_pt
    b3_u = b3 / np.linalg.norm(b3)

    b4 = this_right_pt - this_left_pt 
    b4_u = b4 / np.linalg.norm(b4)

    n_12 = np.cross(b1, b2)
    n_23 = np.cross(b2, b3)
    n_34 = np.cross(b3, b4)

    in1 = np.dot(np.cross(n_12, n_23), b2_u)
    in2 = np.dot(n_12, n_23)

    ala = np.arctan2(in1, in2) / np.pi

    # evan ax
    b1 = last_left_pt - this_left_pt
    b1_u = b1 / np.linalg.norm(b1)
    
    b2 = last_right_pt - last_left_pt
    b2_u = b2 / np.linalg.norm(b2)
    
    b3 = this_right_pt - last_right_pt
    b3_u = b3 / np.linalg.norm(b3)
    
    b4 = this_left_pt - this_right_pt
    b4_u = b4 / np.linalg.norm(b4)
    
    n_12 = np.cross(b1, b2)
    n_23 = np.cross(b2, b3)
    n_34 = np.cross(b3, b4)
    
    in1 = np.dot(np.cross(n_23, n_34), b3_u)
    in2 = np.dot(n_23, n_34)
    
    ela = np.arctan2(in1, in2) / np.pi
    
    x = np.array([cs, ala, ela])
    
    return x 

def get_H_Pairs(data, body_pair_sets, mean_vec_doubles, sigma_inv_doubles):

    cost_dicts = []

    m = len(body_pair_sets)

    for i in range(m-1):

        cost_dict = {}

        current_pairs = body_pair_sets[i]
        num_current_pairs = len(current_pairs)

        next_pairs = body_pair_sets[i+1]
        num_next_pairs = len(next_pairs)

        mean_vec = mean_vec_doubles[i]
        sigma_inv = sigma_inv_doubles[i]

        for j in range(num_current_pairs):

            current_pair = current_pairs[j]
            cp_l, cp_r = current_pair

            for k in range(num_next_pairs):

                next_pair = next_pairs[k]
                np_l, np_r = next_pair

                lattice = (current_pair, next_pair)
                cells = [a for b in lattice for a in b]

                t1 = len(set(cells)) == 4

                if t1:

                    x = get_x_degree_4(lattice, data)
                    cost_dict[lattice] = get_m_dist_Pairs(x, mean_vec, sigma_inv)

        cost_dicts.append(cost_dict)

    return cost_dicts

class Pairs(Exact_HGM):

    def __init__(self, data, H, connections, best_cost, source, start_time, time_limit, num_save, degree_6_means, degree_6_sigma_invs, best_seqs, best_costs):

        self.degree_6_means = degree_6_means
        self.degree_6_sigma_invs = degree_6_sigma_invs

        Exact_HGM.__init__(self, data, H, connections, best_cost, source, start_time, time_limit, num_save, best_seqs, best_costs)

        self.I_cost = np.empty(self.m-2)

    def Branch(self, i):
        
        self.elapsed_time = time.time() - self.start_time
        this_vert = self.seq[i]
        self.Enqueue(i)

        while self.n_Q[i] > 0 and self.elapsed_time <= self.time_limit:

            next_vert = self.get_top(i)
            self.seq_append(next_vert)
            this_H_cost = self.H[i][self.seq[-2], self.seq[-1]]
            self.H_cost[i] = this_H_cost
            self.cost += this_H_cost

            if i >= 1:

                this_triple = get_x_degree_6(self.seq, self.data)
                this_triples_cost = get_m_dist(this_triple, self.degree_6_means[i-1], self.degree_6_sigma_invs[i-1])
                self.I_cost[i-1] = this_triples_cost
                self.cost += this_triples_cost

            i+=1

            if i < self.m - 1:

                self.Branch(i)

            if i == self.m - 1:

                self.cost = self.H_cost.sum() + self.I_cost.sum() 

                if self.cost <= self.cost_star:

                    self.seq_star = self.seq.copy()
                    self.cost_star = self.cost

                    print(self.seq_star, np.round(self.cost_star,2),'- Runtime:', np.round(self.elapsed_time,2), 'seconds')

                if self.cost <= self.best_costs[-1]:

                    self.lattice_storage()


            self.remove_specific(i-1, next_vert)
            self.seq_remove()
            self.H_cost[i-1] = 0
            
            if i >= 2:

                self.I_cost[i-2] = 0
            
            self.cost = self.H_cost.sum() + self.I_cost.sum() 

            i += -1

    def Begin_Search(self):

        i = 0
    
        self.seq_append(self.source)
        self.Branch(i)

# Full

def get_x_doubles_full(lattice, data):

    j = -2

    this_pair = lattice[j]
    this_left, this_right = this_pair
    tl_p = data[this_left]
    tr_p = data[this_right]
    tp_mp = .5*(tl_p + tr_p)

    next_pair = lattice[j+1]
    next_left, next_right = next_pair
    nl_p = data[next_left]
    nr_p = data[next_right]
    np_mp = .5*(nl_p + nr_p)

    r1 = nl_p - tl_p
    r1_u = r1 / np.linalg.norm(r1)

    r2 = nr_p - tr_p
    r2_u = r2 / np.linalg.norm(r2)

    r = np.dot(r1_u, r2_u)

    cos_sim = r


    # ala sum

    lp_left = tl_p
    cp_left = nl_p

    lp_right = tr_p
    cp_right = nr_p

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
    
    angle = np.arctan2(in1, in2) / np.pi

    ala = angle

    # ea sum

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
    
    angle = np.arctan2(in1, in2) / np.pi

    ea = angle

    features = np.array([cos_sim, ala, ea])

    return features

def get_x_triples_full(lattice, data):

    j = -2

    last_pair = lattice[j-1]
    last_left, last_right = last_pair
    ll_p = data[last_left]
    lr_p = data[last_right]
    lp_mp = .5*(ll_p + lr_p)

    this_pair = lattice[j]
    this_left, this_right = this_pair
    tl_p = data[this_left]
    tr_p = data[this_right]
    tp_mp = .5*(tl_p + tr_p)

    next_pair = lattice[j+1]
    next_left, next_right = next_pair
    nl_p = data[next_left]
    nr_p = data[next_right]
    np_mp = .5*(nl_p + nr_p)

    pd_ratio = (np.linalg.norm(nr_p - nl_p) / np.linalg.norm(tr_p - tl_p))
    lat_length = np.linalg.norm(np_mp - tp_mp)

    r1 = lp_mp - tp_mp
    r1_mag = np.linalg.norm(r1)

    r2 = np_mp - tp_mp
    r2_mag = np.linalg.norm(r2)

    new_ray = np.dot(r1, r2)
    new_ray_norm = new_ray / (r1_mag * r2_mag)
    new_ang = (180/np.pi)*np.arccos(new_ray_norm)      

    # plane angle sum

    n1 = np.cross(tr_p - tl_p, tp_mp - lp_mp)
    n1_u = n1 / np.linalg.norm(n1)

    n2 = np.cross(tr_p - tl_p, tp_mp - np_mp)
    n2_u = n2 / np.linalg.norm(n2)

    r_norm = np.dot(n1_u, n2_u)
    angle = (180/np.pi)*np.arccos(r_norm)

    plane_angle = angle

    x = np.array([pd_ratio, lat_length, new_ang, plane_angle])

    return x

def get_x_seq_exp(lattice, data):

    seq_len = len(lattice)

    # Doubles
    cos_sim_sum = 0
    ala_sum = 0
    ea_sum = 0

    # Triples. 
    pd_ratio_sum = 0
    lat_length = 0
    angle_sum = 0
    plane_angle_sum = 0
    
    # pd_ratio_sum + lat_length + cos sim
    for j in range(seq_len-1):

        this_pair = lattice[j]
        this_left, this_right = this_pair
        tl_p = data[this_left]
        tr_p = data[this_right]
        tp_mp = .5*(tl_p + tr_p)

        next_pair = lattice[j+1]
        next_left, next_right = next_pair
        nl_p = data[next_left]
        nr_p = data[next_right]
        np_mp = .5*(nl_p + nr_p)

        if j>0:
            lat_length += np.linalg.norm(np_mp - tp_mp)
            pd_ratio_sum += (np.linalg.norm(nr_p - nl_p) / np.linalg.norm(tr_p - tl_p))

        r1 = nl_p - tl_p
        r1_u = r1 / np.linalg.norm(r1)

        r2 = nr_p - tr_p
        r2_u = r2 / np.linalg.norm(r2)

        r = np.dot(r1_u, r2_u)

        cos_sim_sum += r


        # ala sum

        lp_left = tl_p
        cp_left = nl_p

        lp_right = tr_p
        cp_right = nr_p

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
        
        angle = np.arctan2(in1, in2) / np.pi

        ala_sum += angle

        # ea sum

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
        
        angle = np.arctan2(in1, in2) / np.pi

        ea_sum += angle

    # angle sum
    for j in range(1, seq_len-1):


        last_pair = lattice[j-1]
        last_left, last_right = last_pair
        ll_p = data[last_left]
        lr_p = data[last_right]
        lp_mp = .5*(ll_p + lr_p)

        this_pair = lattice[j]
        this_left, this_right = this_pair
        tl_p = data[this_left]
        tr_p = data[this_right]
        tp_mp = .5*(tl_p + tr_p)

        next_pair = lattice[j+1]
        next_left, next_right = next_pair
        nl_p = data[next_left]
        nr_p = data[next_right]
        np_mp = .5*(nl_p + nr_p)

        r1 = lp_mp - tp_mp
        r1_mag = np.linalg.norm(r1)

        r2 = np_mp - tp_mp
        r2_mag = np.linalg.norm(r2)

        new_ray = np.dot(r1, r2)
        new_ray_norm = new_ray / (r1_mag * r2_mag)
        new_ang = (180/np.pi)*np.arccos(new_ray_norm) 

        angle_sum += new_ang        

        # plane angle sum

        n1 = np.cross(tr_p - tl_p, tp_mp - lp_mp)
        n1_u = n1 / np.linalg.norm(n1)

        n2 = np.cross(tr_p - tl_p, tp_mp - np_mp)
        n2_u = n2 / np.linalg.norm(n2)

        r_norm = np.dot(n1_u, n2_u)
        angle = (180/np.pi)*np.arccos(r_norm)

        plane_angle_sum += angle

    x = np.array([cos_sim_sum, ala_sum, ea_sum, pd_ratio_sum, lat_length, angle_sum, plane_angle_sum])

    return x

class Full(Exact_HGM):

    def __init__(self, data, H, connections, best_cost, source, start_time, time_limit, num_save, seq_means, seq_sigma_invs, mean_cost_est, sigma_inv_cost_est, best_seqs, best_costs):

        self.seq_means = seq_means
        self.seq_sigma_invs = seq_sigma_invs
        self.mean_cost_est = mean_cost_est
        self.sigma_inv_cost_est = sigma_inv_cost_est

        Exact_HGM.__init__(self, data, H, connections, best_cost, source, start_time, time_limit, num_save, best_seqs, best_costs)

        self.total_doubles_features = np.zeros((self.m-1, 3))
        self.total_triples_features = np.zeros((self.m-2, 4))

        self.seq_means_full = seq_means[-1]
        self.seq_sigma_invs_full = seq_sigma_invs[-1]

    def Prune(self, i):

        # First prune for assignment constraints:
        P_next = self.P[i].copy()
        seq = self.seq.copy()

        used_detections = [a for b in seq for a in b]
        Q_next = [a for a in P_next if a[0] not in used_detections and a[1] not in used_detections]

        if i >= 1 and i < self.m - 2:

            # Check for lattice intersection:
            # Q_next = [a for a in Q_next if self.lattice_intersection(a)]

            # Of remaining prune for estimated cost:
            this_seq_means = self.seq_means[i-1]
            this_seq_sigma_invs = self.seq_sigma_invs[i-1]

            mean_cost_est = self.mean_cost_est[i-1]
            sigma_inv_cost_est = self.sigma_inv_cost_est[i-1]

            for a in Q_next:
                seq_temp = seq + [a]
                est_x_seq = get_x_seq_exp(seq_temp, self.data)
                est_cost = get_m_dist(est_x_seq, this_seq_means, this_seq_sigma_invs)
                log_est_cost = np.log(est_cost)

                standardized_log_est_cost = sigma_inv_cost_est*(log_est_cost - mean_cost_est)**2

                if standardized_log_est_cost > 4.5*np.log(self.cost_star):

                    Q_next.remove(a)

        self.Q[i] = Q_next
        self.n_Q[i] = len(Q_next)

    def Branch(self, i):
        
        self.elapsed_time = time.time() - self.start_time
        self.Enqueue(i)

        while self.n_Q[i] != 0 and self.elapsed_time <= self.time_limit:

            next_vert = self.get_top(i)
            self.seq_append(next_vert)
            self.total_doubles_features[i] = get_x_doubles_full(self.seq, self.data)

            if i >= 1:

                self.total_triples_features[i-1] = get_x_triples_full(self.seq, self.data)

            i+=1

            if i < self.m - 1:

                self.Branch(i)

            if i == self.m - 1:

                # sequence cost
                x_seq_doubles = self.total_doubles_features.sum(axis=0)
                x_seq_triples = self.total_triples_features.sum(axis=0)
                x_seq = np.concatenate([x_seq_doubles, x_seq_triples])

                sequence_cost = get_m_dist(x_seq, self.seq_means_full, self.seq_sigma_invs_full)

                self.cost = sequence_cost

                if (self.cost < self.cost_star):

                    self.seq_star = self.seq.copy()
                    self.cost_star = self.cost

                    print(self.seq_star, np.round(self.cost_star,2),'- Runtime:', np.round(self.elapsed_time,2), 'seconds')

                if self.cost <= self.best_costs[-1]:

                    self.lattice_storage()

                self.cost = 0 

            self.remove_specific(i-1, next_vert)
            self.seq_remove()

            # if i <= self.m - 2:

            #     self.clear_queue(i)

            i += -1

    def Begin_Search(self):

        i = 0
    
        self.seq_append(self.source)
        self.Branch(i)

# P-F

class PF(Exact_HGM):

    def __init__(self, data, H, connections, best_cost, source, start_time, time_limit, num_save, degree_6_means, degree_6_sigma_invs, seq_means_full, seq_sigma_invs_full, best_seqs, best_costs):

        self.degree_6_means = degree_6_means
        self.degree_6_sigma_invs = degree_6_sigma_invs
        self.seq_means_full = seq_means_full
        self.seq_sigma_invs_full = seq_sigma_invs_full

        Exact_HGM.__init__(self, data, H, connections, best_cost, source, start_time, time_limit, num_save, best_seqs, best_costs)

        self.I_cost = np.empty(self.m-2)

    def Branch(self, i):
        
        self.elapsed_time = time.time() - self.start_time
        this_vert = self.seq[i]
        self.Enqueue(i)

        while self.n_Q[i] > 0 and self.elapsed_time <= self.time_limit:

            next_vert = self.get_top(i)
            self.seq_append(next_vert)
            this_H_cost = self.H[i][self.seq[-2], self.seq[-1]]
            self.H_cost[i] = this_H_cost
            self.cost += this_H_cost

            if i >= 1:

                this_triple = get_x_degree_6(self.seq, self.data)
                this_triples_cost = get_m_dist(this_triple, self.degree_6_means[i-1], self.degree_6_sigma_invs[i-1])
                self.I_cost[i-1] = this_triples_cost
                self.cost += this_triples_cost

            i+=1

            if i < self.m - 1:

                self.Branch(i)

            if i == self.m - 1:

                x_seq = get_x_seq_exp(self.seq, self.data)
                sequence_cost = get_m_dist(x_seq, self.seq_means_full, self.seq_sigma_invs_full)

                self.cost = self.H_cost.sum() + self.I_cost.sum() + sequence_cost

                if self.cost <= self.cost_star:

                    self.seq_star = self.seq.copy()
                    self.cost_star = self.cost

                    print(self.seq_star, np.round(self.cost_star,2),'- Runtime:', np.round(self.elapsed_time,2), 'seconds')

                if self.cost <= self.best_costs[-1]:

                    self.lattice_storage()

            self.remove_specific(i-1, next_vert)
            self.seq_remove()
            self.H_cost[i-1] = 0
            
            if i >= 2:

                self.I_cost[i-2] = 0
            
            self.cost = self.H_cost.sum() + self.I_cost.sum() 

            i += -1

    def Begin_Search(self):

        i = 0
    
        self.seq_append(self.source)
        self.Branch(i)

