import numpy as np
from .storage import *

from hepqpr.qallse.data_wrapper import * 
from time import clock


def doublet_making(constants, table, detModel, dataw: DataWrapper, time_event = True, debug = True):
	
	nHits = table.shape[0]
	nPhiSlices = constants.nPhiSlices
	nLayers = constants.nLayers
	
	
	if time_event:
		start = clock()
	
	
	def get_layer_range():
		'''
		This function returns the list of layers that contain interesting hits, given our
		chosen inner hit. It also returns the min/max bound for interesting hits for each range.
		'''
		#Construct layer geometries
		layers = np.arange(nLayers, dtype='int8')
		layerGeos = np.array([detModel.layers[i] for i in layers])
		refCoords = np.array([geo.refCoord for geo in layerGeos])

		#Filter layers that are too far away from the inner hit
		layers    = layers[np.abs(refCoords - inner_hit['r']) < constants.maxDoubletLength]
		layerGeos = layerGeos[np.abs(refCoords - inner_hit['r']) < constants.maxDoubletLength]
		refCoords = refCoords[np.abs(refCoords - inner_hit['r']) < constants.maxDoubletLength]


		#Find the bounds of interest for each remaining layer
		maxInterest = constants.zMinus + refCoords * (inner_hit['z'] - constants.zMinus) / inner_hit['r']
		minInterest = constants.zPlus  + refCoords * (inner_hit['z'] - constants.zPlus ) / inner_hit['r']
		for idx in range(len(maxInterest)):
			if minInterest[idx] > maxInterest[idx]:
				minInterest[idx], maxInterest[idx] = maxInterest[idx], minInterest[idx]

		#Filter layers whose bounds of intrest fall outside their geometric bounds 
		mask = [(layerGeos[idx].maxBound > minInterest[idx] and layerGeos[idx].minBound < maxInterest[idx]) for idx in range(len(layers))]
		layers      = layers[mask]
		minInterest = minInterest[mask]
		maxInterest = maxInterest[mask]

		return layers, maxInterest, minInterest
		
	
	#Filter functions
	filter_layers = lambda row: row['layer_id'] in layers
	filter_phi = lambda row: ((row['phi_id'] - 1) == inner_hit['phi_id'] or 
	                          (row['phi_id'] + 1) == inner_hit['phi_id'] or 
	                           row['phi_id'] == inner_hit['phi_id'] or
	                          (row['phi_id'] == 0 and inner_hit['phi_id'] == nPhiSlices - 2) or
	                          (row['phi_id'] == nPhiSlices - 2 and inner_hit['phi_id'] == 0))
	filter_doublet_length = lambda row: ((row['r'] - inner_hit['r']) < constants.maxDoubletLength) & ((row['r'] - inner_hit['r']) > constants.minDoubletLength)
	filter_horizontal_doublets = lambda row: np.abs((row['z'] - inner_hit['z'])/(row['r'] - inner_hit['r'])) < constants.maxCtg
	filter_boring_hits = lambda row: row['z'] > minInterest[np.where(layers == row['layer_id'])] and row['z'] < maxInterest[np.where(layers == row['layer_id'])]
	filter_master = lambda row: filter_layers(row) and filter_phi(row) and filter_doublet_length(row) and filter_horizontal_doublets(row) and filter_boring_hits(row)
	
	
	inner, outer = [], []
	for _, inner_hit in table.iterrows():
		
		#print(inner_hit['hit_id'])
		
		layers, maxInterest, minInterest = get_layer_range()
		outer_hit_candidates = table[[filter_master(hit) for _, hit in table.iterrows()]]
		
		for _, outer_hit in outer_hit_candidates.iterrows():
			inner.append(inner_hit['hit_id'])
			outer.append(outer_hit['hit_id'])
       
	
	doublets = pd.DataFrame({'inner': inner, 'outer': outer})
	doublets.drop_duplicates(keep=False, inplace=True)
	
	
	if time_event:
		end = clock() - start
		print(f'RUNTIME: .../seeding/doublet_making.py  - {end} sec')
		
	if debug:
		print('--Grading Doublets--')
		p, r, ms = dataw.compute_score(doublets)
		print(f'Purity: {p * 100}% real doublets')
		print(f'Missing the following {len(ms)} doublets:')
		for miss in ms:
			print('-------------------------------------------------')
			inner = table.loc[table['hit_id'] == miss[0]][0]
			outer = table.loc[table['hit_id'] == miss[1]][0]
			print('InnerHit: ', inner)
			print('OuterHit: ', outer)
			print('filter_layers: ', filter_layers(outer))
			print('filter_phi: ', filter_phi(outer))
			print('filter_doublet_length: ', filter_doublet_length(outer))
			print('filter_horizontal_doublets: ', filter_horizontal_doublets(outer))
			print('-------------------------------------------------')
	
	return doublets

		

def triple_space():
	print()
	print()
	print()



