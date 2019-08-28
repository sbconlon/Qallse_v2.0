from time import time

import cupy as cp
import numpy as np
import pandas as pd

from .storage import *
from hepqpr.qallse.data_wrapper import * 


def doublet_making(constants, spStorage: SpacepointStorage, detModel, doubletsStorage: DoubletStorage, dataw: DataWrapper, test_mode=False):
	
	time_event, debug = True, False
	nHits = spStorage.x.size
	nPhiSlices = len(spStorage.phiSlices)
	nLayers = len(spStorage.phiSlices[0].layerBegin)
	
	def generate_hit_table() -> np.array:
		"""
		This function transfers the information stored in spacepoint storage into a structured numpy array 
		containing the information necessary for doublet making.
		"""
		table = np.zeros(nHits,
						{'names': ['hit_id', 'layer_id', 'phi_id', 'r', 'z'],
						 'formats': ['int64', 'int64', 'int64', 'float64', 'float64']})
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
			
	
	hitTable = generate_hit_table()
	hitTable.setflags(write=False)         #make hitTable immutable
	
	def struct_to_raw(arr):
		return pd.DataFrame(arr).values
	
	rawHitTable = struct_to_raw(hitTable)
	gpu_rawHitTable = cp.array(rawHitTable)
	
	if time_event:
		start = time()
	
	if debug:
		debug_hit_table(hitTable, spStorage)
		
		
	# Slight refactoring to not use global variables
	def get_layer_range(nLayers, detModel, constants, innerHit):
		'''
		This function returns the list of layers that contain interesting hits, given our
		chosen inner hit. It also returns the min/max bound for interesting hits for each range.
		'''
		#Construct layer geometries
		layers = np.arange(nLayers, dtype='int8')
		layerGeos = np.array([detModel.layers[i] for i in layers])
		refCoords = np.array([geo.refCoord for geo in layerGeos])

		#Filter layers that are too far away from the inner hit
		layers    = layers[np.abs(refCoords - innerHit['r']) < constants.maxDoubletLength]
		layerGeos = layerGeos[np.abs(refCoords - innerHit['r']) < constants.maxDoubletLength]
		refCoords = refCoords[np.abs(refCoords - innerHit['r']) < constants.maxDoubletLength]


		#Find the bounds of interest for each remaining layer
		maxInterest = constants.zMinus + refCoords * (innerHit['z'] - constants.zMinus) / innerHit['r']
		minInterest = constants.zPlus  + refCoords * (innerHit['z'] - constants.zPlus ) / innerHit['r']
		for idx in range(len(maxInterest)):
			if minInterest[idx] > maxInterest[idx]:
				minInterest[idx], maxInterest[idx] = maxInterest[idx], minInterest[idx]

		#Filter layers whose bounds of intrest fall outside their geometric bounds 
		mask = [(layerGeos[idx].maxBound > minInterest[idx] and layerGeos[idx].minBound < maxInterest[idx]) for idx in range(len(layers))]
		layers      = layers[mask]
		minInterest = minInterest[mask]
		maxInterest = maxInterest[mask]

		return layers, maxInterest, minInterest
	
	# Filter function
	def filter_table_gpu(hitTable, gpu_rawHitTable, layers, innerHit, constants, nPhiSlices):
		"""
		New filtering method (columnar / vectorized for GPU)
		"""
		filter_layers_mask = np.isin(hitTable[:, 1], layers)
		outerHitSet = gpu_rawHitTable[filter_layers_mask]

		minus_one_mask = (outerHitSet[:, 2] - 1) == innerHit[2]
		plus_one_mask = (outerHitSet[:, 2] + 1) == innerHit[2]
		equals_mask = outerHitSet[:, 2] == innerHit[2]
		dual_mask = (outerHitSet[:, 2] == 0) & (innerHit[2] == nPhiSlices - 2)
		reverse_dual_mask = (outerHitSet[:, 2] == nPhiSlices - 2) & (innerHit[2] == 0)
		filter_phi_mask = minus_one_mask | plus_one_mask | equals_mask | dual_mask | reverse_dual_mask
		outerHitSet = outerHitSet[filter_phi_mask]

		delta_below_max = (outerHitSet[:, 3] - innerHit[3]) < constants.maxDoubletLength
		delta_above_min = (outerHitSet[:, 3] - innerHit[3]) > constants.minDoubletLength
		filter_doublet_length_mask = delta_below_max & delta_above_min
		outerHitSet = outerHitSet[filter_doublet_length_mask]

		filter_horizontal_doublets_mask = cp.abs((outerHitSet[:, 4] - innerHit[4]) / (outerHitSet[:, 3] - innerHit[3]))  < constants.maxCtg
		outerHitSet = outerHitSet[filter_horizontal_doublets_mask]

		gpu_minInterest = cp.array(minInterest)
		gpu_maxInterest = cp.array(maxInterest)
		gpu_layers = cp.array(layers)

		if len(outerHitSet) == 0:
			return outerHitSet

		# substitute the for loop with a matrix op
		x = outerHitSet[:, 1].reshape(-1, 1)            # shape: (row_num, 1)
		gpu_layers = cp.tile(gpu_layers, (len(x), 1))
		index = cp.where(cp.equal(gpu_layers, x))[1]
		f_collection = gpu_minInterest[index]
		g_collection = gpu_maxInterest[index]

		boring_gt_mask = outerHitSet[:, 4] > f_collection
		boring_lt_mask = outerHitSet[:, 4] < g_collection
		filter_boring_hits_mask = boring_gt_mask & boring_lt_mask
		outerHitSet = outerHitSet[filter_boring_hits_mask]

		return outerHitSet
	
	
	
			
	indxCount = 0
	for innerHit in hitTable:
		
		#Get the layers of interest for our inner hit
		layers, maxInterest, minInterest = get_layer_range(nLayers, detModel, constants, innerHit)
		
		# Filter on GPU
		res_new_gpu = filter_table_gpu(rawHitTable[indxCount:], gpu_rawHitTable[indxCount:],
                                   layers, innerHit, constants, nPhiSlices)
		
		for outerHit in res_new_gpu.get(): # bringing to host all at once is the cheapest way
			doubletsStorage.inner.append(innerHit[0])
			doubletsStorage.outer.append(outerHit[0])
			
		indxCount += 1

        
	if time_event:
		runtime = time() - start
		if debug:
			print(f'RUNTIME: .../seeding/doublet_making.py  - {runtime} sec')
		if test_mode:
			doublets = pd.DataFrame({'inner': doubletsStorage.inner, 'outer': doubletsStorage.outer})
			p, r, _ = dataw.compute_score(doublets)
			return [runtime, round(r, 2), round(p, 2), len(doubletsStorage.inner)]
		
	if debug:
		doublets = pd.DataFrame({'inner': doubletsStorage.inner, 'outer': doubletsStorage.outer})
		print('--Grading Doublets--')
		p, r, ms = dataw.compute_score(doublets)
		print(f'Purity: {p * 100}% real doublets')
		print(f'Missing the following {len(ms)} doublets:')
		for miss in ms:
			print('-------------------------------------------------')
			innerHit = hitTable[hitTable['hit_id'] == miss[0]][0]
			outerHit = hitTable[hitTable['hit_id'] == miss[1]][0]
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

