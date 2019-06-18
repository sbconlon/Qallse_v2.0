import numpy as np
from .storage import *

from hepqpr.qallse.data_wrapper import * 
from time import clock


def doublet_making(constants, spStorage: SpacepointStorage, detModel, doubletsStorage: DoubletStorage, dataw: DataWrapper, time_event = True, debug = True):
	
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
	
	if time_event:
		start = clock()
	
	if debug:
		debug_hit_table(hitTable, spStorage)
	
	#Filter functions
	filter_layers = lambda row: row['layer_id'] in layers
	filter_phi = lambda row: ((row['phi_id'] - 1) == innerHit['phi_id'] or 
	                          (row['phi_id'] + 1) == innerHit['phi_id'] or 
	                           row['phi_id'] == innerHit['phi_id'] or
	                          (row['phi_id'] == 0 and innerHit['phi_id'] == nPhiSlices - 2) or
	                          (row['phi_id'] == nPhiSlices - 2 and innerHit['phi_id'] == 0))
	filter_doublet_length = lambda row: ((row['r'] - innerHit['r']) < constants.maxDoubletLength) & ((row['r'] - innerHit['r']) > constants.minDoubletLength)
	filter_horizontal_doublets = lambda row: np.abs((row['z'] - innerHit['z'])/(row['r'] - innerHit['r'])) < constants.maxCtg
	filter_boring_hits = lambda row: row['z'] > minInterest[np.where(layers == row['layer_id'])] and row['z'] < maxInterest[np.where(layers == row['layer_id'])]
	filter_master = lambda row: filter_layers(row) and filter_phi(row) and filter_doublet_length(row) and filter_horizontal_doublets(row) and filter_boring_hits(row)
		
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
			
	indxCount = 0
	for innerHit in hitTable:
		
		#Get the layers of interest for our inner hit
		layers, maxInterest, minInterest = get_layer_range()
		
		'''
		#Filter hits in uninteresting layers
		outerHitSet = np.array([hit for hit in hitTable[indxCount:] if filter_layers(hit)])
		
		#Filter hits in phi slices that are not in +/- one phi slice of the inner hit
		outerHitSet = np.array([hit for hit in outerHitSet if filter_phi(hit)])
		
		#Filter hits that yeild doublets that are too long or too short
		outerHitSet = np.array([hit for hit in outerHitSet if filter_doublet_length(hit)])
		
		#Filter hits that yeild doublets that are too horizontal
		outerHitSet = np.array([hit for hit in outerHitSet if filter_horizontal_doublets(hit)])
		
		#Filter hits outside of the zone of interest
		outerHitSet = np.array([hit for hit in outerHitSet if filter_boring_hits(hit)])
		'''
		
		outerHitSet = np.array([hit for hit in hitTable[indxCount:] if filter_master(hit)])
		
		for outerHit in outerHitSet:
			doubletsStorage.inner.append(innerHit['hit_id'])
			doubletsStorage.outer.append(outerHit['hit_id'])
		
		indxCount += 1
       
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



