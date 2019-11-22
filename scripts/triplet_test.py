#!/usr/bin/env python

import hepqpr.qallse as ql
import pandas as pd

if __name__ == '__main__':
	truth_df = pd.read_csv('20-density-truth.csv')
	hits_df = pd.read_csv('20-density-hits.csv')
	doublets_df = pd.read_csv('20-density-doublets.csv')

	dataw = ql.DataWrapper(hits_df, truth_df)
	model = ql.Qallse(dataw)

	def grade_triplets(tplets):
		total_sample = tplets.shape[0]
		correct = 0
		hits = hits_df.values
		for t in range(tplets.shape[0]):
			if dataw.is_real_xplet([hits[tplets[t, 0]][0],
									hits[tplets[t, 1]][0],
									hits[tplets[t, 2]][0]]):
				correct += 1
		return correct/total_sample

	time_results = []
	ntrials = 1
	for _ in range(ntrials):
		triplets_built = model.build_model(doublets_df, test_mode=True)
		print('Grading Triplets')
		print('Precision: ', grade_triplets(triplets_built))
