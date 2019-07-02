import numpy as np
from .storage import *

from hepqpr.qallse.data_wrapper import * 
from time import clock

from numba import jit, guvectorize
from numba import int32, float32

def doublet_making(constants, spStorage: SpacepointStorage, detModel, doubletsStorage: DoubletStorage, dataw: DataWrapper):
	
	time_event, debug = True, True
	
	nHits = spStorage.x.size
	nPhiSlices = len(spStorage.phiSlices)
	nLayers = len(spStorage.phiSlices[0].layerBegin)
	modelLayers = np.zeros(detModel.layers.shape)
	
	#Copy the class attributes used in get_layer_range to a local variable so that the numba implementation runs
	np.copyto(modelLayers, detModel.layers)				 
	maxDoubletLength = constants.maxDoubletLength
	minDoubletLength = constants.minDoubletLength
	zPlus = constants.zPlus
	zMinus = constants.zMinus
	
	FALSE_INT = 99999    #Integer that reperesents a false value
	
	
	def generate_hit_table() -> np.array:
		"""
		This function transfers the information stored in spacepoint storage into a structured numpy array 
		containing the information necessary for doublet making.
		"""
		table = np.zeros(nHits,
						{'names': ['hit_id', 'layer_id', 'phi_id', 'r', 'z'],
						 'formats': ['int32', 'int32', 'int32', 'float32', 'float32']})
		table['r'] = spStorage.r
		table['z'] = spStorage.z
		table['hit_id'] = spStorage.hit_id
		for phiIdx in range(nPhiSlices):
			for layerIdx in range(nLayers):
				startIdx = spStorage.phiSlices[phiIdx].layerBegin[layerIdx]
				endIdx = spStorage.phiSlices[phiIdx].layerEnd[layerIdx]
				for idx in range(startIdx, endIdx):
					table['layer_id'][idx] = layerIdx
					table['phi_id'][idx] = phiIdx
		return table
			
	
	hit_table = generate_hit_table()
	hit_table.setflags(write=False)         #make hitTable immutable
	
	if debug:
		debug_hit_table(hit_table, spStorage)
	
	
	if time_event:
		start = clock()
	
	
	#Filter functions
	filter_layers = lambda outer_hit, layers: outer_hit['layer_id'] in layers
	
	filter_phi = lambda outer_hit, inner_hit: ((outer_hit['phi_id'] - 1) == inner_hit['phi_id'] or 
	                                     (outer_hit['phi_id'] + 1) == inner_hit['phi_id'] or 
	                                      outer_hit['phi_id'] == inner_hit['phi_id'] or
	                                     (outer_hit['phi_id'] == 0 and inner_hit['phi_id'] == nPhiSlices - 2) or
	                                     (outer_hit['phi_id'] == nPhiSlices - 2 and inner_hit['phi_id'] == 0))
	
	filter_doublet_length = lambda outer_hit, inner_hit: ((outer_hit['r'] - inner_hit['r']) < constants.maxDoubletLength) & ((outer_hit['r'] - inner_hit['r']) > constants.minDoubletLength)
	
	filter_horizontal_doublets = lambda outer_hit, inner_hit: np.abs((outer_hit['z'] - inner_hit['z'])/(outer_hit['r'] - inner_hit['r'])) < constants.maxCtg
	
	filter_boring_hits = lambda outer_hit, layers, minInterest, maxInterest: outer_hit['z'] > minInterest[np.where(layers == outer_hit['layer_id'])] and outer_hit['z'] < maxInterest[np.where(layers == outer_hit['layer_id'])]
	
	filter_master = lambda outer_hit, inner_hit, layers, minInterest, maxInterest: (filter_layers(outer_hit, layers) and 
	                                                                                filter_phi(outer_hit, inner_hit) and 
	                                                                                filter_doublet_length(outer_hit, inner_hit) and 
	                                                                                filter_horizontal_doublets(outer_hit, inner_hit) and 
	                                                                                filter_boring_hits(outer_hit, layers, minInterest, maxInterest))
	                                                                                
	
	
	@guvectorize(["boolean(int32, int32, float32, float32, int32, float32, float32, int32[:], float32[:], float32[:])"],
				  "(),(),(),(),(),(),(),(n),(m),(m)->(n)")
	def filter_numba(o_layer, o_phi, o_r, o_z, i_phi, i_r, i_z, layers, minInterest, maxInterest):
		'''
		This function combines the lambda filters above into one filter and is compiled by numba into a general numpy universal function
		Note: the prefix o_ and i_ in the names of the parameters stand for outer and inner hit respectively.
		'''
		#filter_layers
		if not o_layer in layers:
			return False
			
		#filter_phi
		if not ((o_phi - 1) == i_phi or 
	            (o_phi + 1) == i_phi or 
	             o_phi == i_phi or
	            (o_phi == 0 and i_phi == nPhiSlices - 2) or
	            (o_phi == nPhiSlices - 2 and i_phi == 0)):
			return False
			
		#filter_doublet_length
		if not (((o_r - i_r) < constants.maxDoubletLength) and ((o_r - i_r) > constants.minDoubletLength)):
			return False
			
		#filter_horizontal_doublets
		if not np.abs((o_z - i_z)/(o_r - i_r)) < constants.maxCtg:
			return False
		
		#filter_boring_hits
		if not (o_z > minInterest[np.where(layers == o_layer)] and o_z < maxInterest[np.where(layers == o_layer)]):
			return False
		
		return True



	@jit(nopython=True)
	def get_layer_range(inner_hit):
		'''
		This function returns the list of layers that contain interesting hits, given our chosen inner hit. 
		It also returns the min/max bound in the z-direction for interesting hits for each outer layer.
		'''
		#Construct layer geometries
		layers = np.arange(nLayers)
		refCoords = np.array([modelLayers[layer_idx][1] for layer_idx in layers])

		#Filter layers that are too far away from the inner hit
		layers    = layers[np.abs(refCoords - inner_hit['r']) < maxDoubletLength]
		refCoords = refCoords[np.abs(refCoords - inner_hit['r']) < maxDoubletLength]

		#Find the bounds of interest for each remaining layer
		maxInterest = zMinus + refCoords * (inner_hit['z'] - zMinus) / inner_hit['r']
		minInterest = zPlus  + refCoords * (inner_hit['z'] - zPlus) / inner_hit['r']
		for idx in range(len(maxInterest)):
			if minInterest[idx] > maxInterest[idx]:
				minInterest[idx], maxInterest[idx] = maxInterest[idx], minInterest[idx]

		#Filter layers whose bounds of intrest fall outside their geometric bounds 
		mask = [(modelLayers[layers[idx]][3] > minInterest[idx] and modelLayers[layers[idx]][2] < maxInterest[idx]) for idx in range(len(layers))]
		idx_count = 0
		for bool_idx in mask:
			if not bool_idx:
				layers[idx_count] = FALSE_INT
				maxInterest[idx_count] = FALSE_INT
				minInterest[idx_count] = FALSE_INT
		idx_count += 1

		return layers, maxInterest, minInterest
			
	
	for indx in range(nHits//2):
		
		# If there are an even number of rows, the final iteration should be ignored
		if indx > (nHits-indx-1):
			continue
		
		# If there are an odd number of rows, then the final iteration should only use one row
		if indx == (nHits-indx-1):
			inner_hit = hit_table[indx]
			layers_one, maxInterest_one, minInterest_one = get_layer_range(inner_hit)
			outerHitSet_one = np.array([hit for hit in hitTable[indx:] if filter_numba(hit['layer_id'], hit['phi_id'], hit['r'], hit['z'], 
			                                                                           inner_hit['phi_id'], inner_hit['r'], inner_hit['z'],
			                                                                           layers, minInterest, maxInterest)])
			
			for outerHit in outerHitSet_one:
				doubletsStorage.inner.append(hit_table[indx]['hit_id'])
				doubletsStorage.outer.append(outerHit['hit_id'])
		# Otherwise, two rows should be used
		else:
			inner_hit_one = hit_table[indx]
			inner_hit_two = hit_table[nHits-indx-1]
			
			layers_one, maxInterest_one, minInterest_one = get_layer_range(inner_hit_one)
			layers_two, maxInterest_two, minInterest_two = get_layer_range(inner_hit_two)
			
			outerHitSet_one = np.array([hit for hit in hit_table[indx:] if filter_numba(hit['layer_id'], hit['phi_id'], hit['r'], hit['z'],
			                                                                            inner_hit_one['phi_id'], inner_hit_one['r'], inner_hit_one['z'],
			                                                                            layers_one, minInterest_one, maxInterest_one)])
			outerHitSet_two = np.array([hit for hit in hit_table[nHits-indx-1:] if filter_numba(hit['layer_id'], hit['phi_id'], hit['r'], hit['z'],
			                                                                                    inner_hit_two['phi_id'], inner_hit_two['r'], inner_hit_two['z'], 
			                                                                                    layers_two, minInterest_two, maxInterest_two)])
			
			for outerHit in outerHitSet_one:
				doubletsStorage.inner.append(hit_table[indx]['hit_id'])
				doubletsStorage.outer.append(outerHit['hit_id'])
			for outerHit in outerHitSet_two:
				doubletsStorage.inner.append(hit_table[nHits-indx-1]['hit_id'])
				doubletsStorage.outer.append(outerHit['hit_id'])
		
				
	if time_event:
		end = clock() - start
		print(f'RUNTIME: .../seeding/doublet_making.py  - {end} sec')
		
		
		
		
		
	if debug:
		doublets = pd.DataFrame({'inner': doubletsStorage.inner, 'outer': doubletsStorage.outer})
		print('--Grading Doublets--')
		p, r, ms = dataw.compute_score(doublets)
		print(f'Purity: {p * 100}% real doublets')
		print(f'Missing the following {len(ms)} doublets:')
		for miss in ms:
			print('-------------------------------------------------')
			innerHit = hit_table[hit_table['hit_id'] == miss[0]][0]
			outerHit = hit_table[hit_table['hit_id'] == miss[1]][0]
			print('InnerHit: ', innerHit)
			print('OuterHit: ', outerHit)
			print('filter_layers: ', filter_layers(outerHit))
			print('filter_phi: ', filter_phi(outerHit))
			print('filter_doublet_length: ', filter_doublet_length(outerHit))
			print('filter_horizontal_doublets: ', filter_horizontal_doublets(outerHit))
			print('-------------------------------------------------')



def debug_hit_table(table, storage):
	'''
	This function iterates through the chunks and test if the hitTable and spStorage objects
	contain the same number of hits in each chunk.
	'''
	result = []
	for phiIdx in range(len(storage.phiSlices)):
		for layerIdx in range(len(storage.phiSlices[0].layerBegin)):
			hitCount = 0 
			for row in table:
				if row['phi_id'] == phiIdx and row['layer_id'] == layerIdx:
					hitCount += 1
			hitsInStorage = storage.phiSlices[phiIdx].layerEnd[layerIdx] - storage.phiSlices[phiIdx].layerBegin[layerIdx]
			result.append(hitCount == hitsInStorage)
			hitCount = 0
	print('Hit Table Debug: ', all(result))

		

def triple_space():
	print()
	print()
	print()



