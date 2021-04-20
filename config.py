
# Exact Hypergraph Matching for Nuclei Identification in Embryonic C. elegans
# Configuration

import os

'''
1) Preset Model choice
 - 'QAP' uses a quadratic objective function and attempts to solve nuclear identification as graph matching.
 - 'Pairs' uses degree four and six hyperedges to better model local nuclear relationships.
 - 'Full' uses exclusively degree n features to cost the full assignment.
 - 'PF' uses both feature sets from Pairs and Full.

2) Strain: Six C. elegans strains are imaged: 'RW10598', 'RW10896', 'KP9305', 'RW10131', 'RW10752', 'JCC596'
   Position: Multiple embryos are imaged for each strain. Each have different names corresponding to the 
   location and date of imaging.

	Choose an imaged worm in which to run Exact_HGM:

	RW10598:
	 - 0: 'Pos0'
	 - 1: '598_Slitscan_6um_5min_Pos1'
	 - 2: 'Pos4-bgsub_95_iteration_5'

	RW10896:
	 - 3: 'Pos1'
	 - 4: 'Pos4'
	 - 5: 'Pos6'

	KP9305
	 - 6: 'Pos0'
	 - 7: 'Pos2'
	 - 8: 'Pos4'

	RW10131
	 - 9: '052918_Pos1'

	RW10752
	 - 10: '0225_Pos0'
	 - 11: '0312_Pos1'
	 - 12: '0312_Pos2'

	JCC596
	 - 13: '082619_Pos3'
	 - 14: '091119_Pos2'
	 - 15: '091119_Pos3'

'''

model = 'Pairs'
dataset = 10

# 4) Data range: Pick a sequence of images from the selected dataset in which to run Exact_HGM:
start_idx = 0
end_idx = 100

# 5) Seed nuclei
# Choose nuclei indices 0, ..., n-1 to assume given as input. An empty list will initiate the unrestricted search
seed_nuclei = [18,19,16,17,14,15]

# 6) Name the experiment. This is just for organizing results
experiment_name = 'T-V5'

# 6) Number of top assignments to save
num_save = 5

# 7) Maximum runtime (in minutes!)
time_limit_minutes = 100 

#8) Upper bound on global minimum
initial_cost = 1e6

# Save configuration:
config = {}
config['model'] = model
config['dataset'] = dataset
config['start_idx'] = max(start_idx, 0)
config['end_idx'] = end_idx
config['seed_nuclei'] = seed_nuclei
config['experiment_name'] = experiment_name
config['home'] = os.path.split(os.getcwd())[0]
config['num_save'] = num_save
config['time_limit'] = time_limit_minutes * 60
config['initial_cost'] = initial_cost
