import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import glob
from math import hypot

if __name__ == '__main__':
	cori_results_path = r'/home/atlas/Qallse_v2/Qallse_v2.0/scripts/cori_results/'
	pc_results_path = r'/home/atlas/Qallse_v2/Qallse_v2.0/scripts/pc_results/'
	cori_files = glob.glob(cori_results_path + '/*.csv')
	pc_files = glob.glob(pc_results_path + '/*.csv')
	
	for results_folder in [('PC Results', pc_files), ('Cori Results', cori_files)]:
		graph_name = results_folder[0]
		densities = []
		runtimes = []
		deviations = []
		for filename in results_folder[1]:
			df = pd.read_csv(filename)
			densities.append(df.at[0, 'density'])
			runtimes.append(df['doublets_made'].mean() / df['runtime'].mean())
			deviations.append(hypot(df['runtime'].std()/df['runtime'].mean(), df['doublets_made'].std()/df['doublets_made'].mean()))
		print(deviations)
		plt.figure()
		plt.errorbar(np.array(densities), np.array(runtimes), yerr= np.array(deviations), fmt='o', ecolor='g')
		plt.title(graph_name)
		
	plt.show()
