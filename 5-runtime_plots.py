import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import glob
from math import hypot

if __name__ == '__main__':
	
	plot_colors = {'Parallel': 'blue', 'Numpy': 'red', 'Original': 'green'}
	plot_groups = {'Parallel': 'Parallel Method', 'Numpy': 'GUvectorize Method', 'Original': 'Loop Method'}
	
	for hardware in ['Cori', 'PC']:
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1, xlabel='Number of Hits', ylabel='Runtime (sec)')
		ax.set_yscale('log')
		plt.title('Seeding Runtime Comparison on {}'.format(hardware))
		for algorithm in ['Parallel', 'Numpy', 'Original']:
			input_path = r'/home/atlas/Qallse_v2/{}_{}_results/'.format(algorithm.lower(), hardware.lower())
			files = glob.glob(input_path + '/*.csv')
			densities = []
			runtimes = []
			deviations = []
			for filename in files:
				df = pd.read_csv(filename)
				densities.append(df.at[0, 'density'] * 56620)
				runtimes.append(df['runtime'].mean())
				deviations.append(df['runtime'].std())
			plt.errorbar(np.array(densities), np.array(runtimes), yerr= np.array(deviations), 
						 fmt='o', ecolor=plot_colors[algorithm], c=plot_colors[algorithm], label=plot_groups[algorithm])
		plt.legend(loc=2)
		
	plt.show()
