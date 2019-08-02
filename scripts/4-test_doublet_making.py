#!/usr/bin/env python 
'''
This script tests the performance of the doublet making algorithm
'''

import pandas as pd
from hepqpr.qallse.dsmaker import create_dataset

if __name__ == '__main__':
	
	#---- Script Constants
	test_densities = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
	trials_per_density = 1
	input_seed = None
	
	#---- Test Doublet Making
	results, indx = [[i] for i in test_densities], 0
	for ds in test_densities:
		for _ in range(trials_per_density):
			print(f'--> Testing doublet_making with {ds} density')
			stat_row = create_dataset(density=ds,
									  min_hits_per_track=5,
									  high_pt_cut=1.0,
									  random_seed=input_seed,
									  double_hits_ok=False,
									  gen_doublets=True,
									  test_mode=True)
			results[indx].extend(stat_row)
			indx += 1
	
	#---- Write Tests to Disk
	titles = 'density,runtime,recall,precision,doublets_made,seed'.split(',')
	stats = pd.DataFrame(results, columns=titles)
	print('--> Stats')
	print(stats)
	stats.to_csv('parallel_method_results.csv', index=False)
		
		
			
