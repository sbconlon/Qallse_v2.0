#!/usr/bin/env python 
'''
This script tests the performance of the doublet making algorithm
'''

import pandas as pd
from hepqpr.qallse.dsmaker import create_dataset

if __name__ == '__main__':
	
	#---- Script Constants
	test_densities = [1] * 3
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
	stats.to_csv('100_numpy_results.csv', index=False)
		
		
			
