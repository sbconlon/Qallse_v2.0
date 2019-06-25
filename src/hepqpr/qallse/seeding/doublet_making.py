import numpy as np
from .storage import *

from hepqpr.qallse.data_wrapper import * 
from time import clock


def doublet_making(constants, table, detModel, dataw: DataWrapper, z_map, time_event = True, debug = True):
	
	nHits = table.shape[0]
	nPhiSlices = constants.nPhiSlices
	nLayers = constants.nLayers
	maxCtg = constants.maxCtg
	maxDoubletLength = constants.maxDoubletLength
	minDoubletLength = constants.minDoubletLength
	
	if time_event:
		start = clock()
	
	
	def get_index_ranges():
		'''
		This function returns the list of layers that contain interesting hits, given our
		chosen inner hit. It also returns the min/max bound for interesting hits for each range.
		'''
		#Construct layer geometries
		layer_range = np.arange(2, nLayers-2, dtype='int8')
		layerGeos = np.array([detModel.layers[i-2] for i in layer_range])
		refCoords = np.array([geo.refCoord for geo in layerGeos])

		#Filter layers that are too far away from the inner hit
		layer_range   = layer_range[np.abs(refCoords - inner_r) < constants.maxDoubletLength]
		layerGeos = layerGeos[np.abs(refCoords - inner_r) < constants.maxDoubletLength]
		refCoords = refCoords[np.abs(refCoords - inner_r) < constants.maxDoubletLength]


		#Find the bounds of interest for each remaining layer
		maxInterest = constants.zMinus + refCoords * (inner_z - constants.zMinus) / inner_r
		minInterest = constants.zPlus  + refCoords * (inner_z - constants.zPlus ) / inner_r
		for idx in range(len(maxInterest)):
			if minInterest[idx] > maxInterest[idx]:
				minInterest[idx], maxInterest[idx] = maxInterest[idx], minInterest[idx]


		#Filter layers whose bounds of intrest fall outside their geometric bounds 
		mask = [(layerGeos[idx].maxBound > minInterest[idx] and layerGeos[idx].minBound < maxInterest[idx]) for idx in range(len(layer_range))]
		layer_range = layer_range[mask]
		minInterest = minInterest[mask]
		maxInterest = maxInterest[mask]
		
		
		#Set the z range to the z_ids within the min and max z_id range
		z_range = dict(list(zip(layer_range, [[z_map(minInterest[i]), z_map(maxInterest[i])] for i in range(len(minInterest))])))
		
		
		#Set the phi range to the phi slices within plus or minus one from the inner phi slice
		if inner_phi_idx == 0:
			phi_range = [nPhiSlices-2, 0, 1]
		elif inner_phi_idx == nPhiSlices-2:
			phi_range = [nPhiSlices-3, nPhiSlices-2, 0]
		else:
			phi_range = [inner_phi_idx-1, inner_phi_idx, inner_phi_idx+1]
		
		return layer_range, phi_range, z_range
		
	
	'''
	#Filter strings
	filter_layers = 'layer_id in @layers'
	filter_phi = '(phi_id - 1) == @inner_phi | (phi_id + 1) == @inner_phi | phi_id == @inner_phi | (phi_id == 0 & @inner_phi == @nPhiSlices - 2) | (phi_id == @nPhiSlices - 2 & @inner_phi == 0)'
	filter_doublet_length = '((r - @inner_r) < @maxDoubletLength) & ((r - @inner_r) > @minDoubletLength)'
	filter_horizontal_doublets = '((z - @inner_z)/(r - @inner_r)) < @maxCtg'
	filter_boring_hits = '(z > @minInterest[layer_id]) & (z < @maxInterest[layer_id])'
	plus = ' & '
	filter_one = filter_layers + plus + filter_phi + plus + filter_doublet_length + plus + filter_horizontal_doublets
	filter_all = filter_layers + plus + filter_phi + plus + filter_doublet_length + plus + filter_horizontal_doublets + plus + filter_boring_hits
	'''
	
	inner, outer = [], []
	for _, inner_hit in table.iterrows():
		
		inner_layer_idx = inner_hit.name[0]
		inner_phi_idx = inner_hit.name[1]
		inner_z_idx = inner_hit.name[2]
		inner_hit_id = inner_hit['hit_id']
		inner_r = inner_hit['r']
		inner_z = inner_hit['z']
		
		layer_range, phi_range, z_range = get_index_ranges()
		
		outer_hit_candidates = table.drop(columns=['x', 'z', 'r'])
		
		#Filter indexes outside the valid layer range
		ix = outer_hit_candidates.index.get_level_values('layer_id').isin(layer_range)
		outer_hit_candidates = outer_hit_candidates[ix]
		
		#Filter indexes outside the valid phi range
		ix = outer_hit_candidates.index.get_level_values('phi_id').isin(phi_range)
		outer_hit_candidates = outer_hit_candidates[ix]
		
		#Filter indexes inside the z range
		z_filter = []
		for _, outer_hit in outer_hit_candidates.iterrows():
			outer_layer_idx = outer_hit.name[0]
			outer_z_idx = outer_hit.name[2]
			z_filter.append(outer_z_idx in list(range(z_range[outer_layer_idx][0], z_range[outer_layer_idx][1])))
		outer_hit_candidates = outer_hit_candidates[z_filter]
		
		#Create doublets
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
		'''
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
		'''
	
	return doublets

		

def triple_space():
	print()
	print()
	print()
	
	




