#!/usr/bin/env python

import hepqpr.qallse as ql
import pandas as pd
import numpy as np

if __name__ == '__main__':
	truth_df = pd.read_csv('event000001000-truth.csv')
	hits_df = pd.read_csv('event000001000-hits.csv')
	doublets_df = pd.read_csv('event000001000-doublets.csv')
	
	dataw = ql.DataWrapper(hits_df, truth_df)
	model = ql.Qallse(dataw)
	
	time_results = []
	ntrials = 1
	for _ in range(ntrials):
		i = model.build_model(doublets_df, test_mode=True, compare=False)
		#print(i)
		#time_results.append(i)
		
	#print(time_results)
	
	#pd.DataFrame(np.array(time_results)).to_csv('build_model_times.csv', index=False)
	
	
