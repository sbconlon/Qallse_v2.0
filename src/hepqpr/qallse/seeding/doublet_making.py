import numpy as np
from .storage import *

from time import clock


def doublet_making(constants, spStorage: SpacepointStorage, detModel, doubletsStorage: DoubletStorage, time_event = True):
	if time_event:
		start = clock()
	
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
	
	print(hitTable)
	
	#debug_hit_table(hitTable, spStorage)
	
	'''
	def area_of_interest(r, z, layerID):
		
		This function returns the range of interest for a given layer and coordinates of inner point.
		Or, returns false tuple if min or max coordinate is out of bounds 
		The formulas for this function was taken from lines 85 and 86 of doublet_making.py in Qallse
		
		layerGeo = detModel.layers[layerID]
		refCoord = layerGeo.refCoord
		minCoord = constants.zMinus + refCoord * (z - constants.zMinus) / r
		maxCoord = constants.zPlus + refCoord * (z - constants.zPlus) / r
		if minCoord > maxCoord:
			minCoord, maxCoord = maxCoord, minCoord
		if layerGeo.maxBound < minCoord or layerGeo.minBound > maxCoord:
			return (False, False)
		return (minCoord, maxCoord)
	'''		
	
	for innerHit in hitTable:
		
		#Filter hits in layers less than the layer of the innerHit and only allow hits in the same
		#or to the right of the innerHit's phi slice.
		filter_layers = lambda row: row['layer_id'] > innerHit['layer_id']
		filter_phi = lambda row: ((row['phi_id'] + 1 % nPhiSlices) == innerHit['phi_id'] or row['phi_id'] == innerHit['phi_id'])
		outerHitSet = np.array([hit for hit in hitTable[innerHit['hit_id']:] if (filter_layers(hit) and filter_phi(hit))])
	
		
		'''print(f'outerHitSet ({len(outerHitSet)}) for ({innerHit}) after CUT 1: ')
		print(outerHitSet)
		triple_space()'''
		
		inR, inZ = innerHit['r'], innerHit['z']
		
		'''
		#Filter hits that belong to layers outside the area of interest
		def filter_area_of_interest(row):
			
			This function filters all hits in layers outside of the area of interest for the given layer.
			
			minCoord, maxCoord = area_of_interest(inR, inZ, row['layer_id'])
			if minCoord:
				return row['r'] > minCoord and row['r'] < maxCoord
			return False
		outerHitSet = np.array([hit for hit in outerHitSet if filter_area_of_interest(hit)])
		
		
		print(f'outerHitSet for ({innerHit}) after CUT 2: ')
		print(outerHitSet)
		triple_space()
		
		
		#Filter hits that are outside the area of interest
		def filter_hits_of_interest(row):
			minCoord, maxCoord = area_of_interest(inR, inZ, row['layer_id'])
			if not(minCoord is False):
				return row['r'] < maxCoord and row['r'] > minCoord
			return False
		outerHitSet = np.array([hit for hit in outerHitSet if filter_hits_of_interest(hit)])
		
		print(f'outerHitSet ({len(outerHitSet)}) for ({innerHit}) after CUT 3: ')
		print(outerHitSet)
		triple_space()
		'''
		
		#Filter hits that yeild doublets that are too long or too short
		filter_doublet_length = lambda row: ((row['r'] - inR) < constants.maxDoubletLength and
		                                      (row['r'] - inR) > constants.minDoubletLength)
		outerHitSet = np.array([hit for hit in outerHitSet if filter_doublet_length(hit)])
		
		'''print(f'outerHitSet ({len(outerHitSet)}) for ({innerHit}) after CUT 4: ')
		print(outerHitSet)
		triple_space()'''
		
		#Filter hits that yeild doublets that are too horizontal
		filter_horizontal_doublets = lambda row: np.abs((row['z'] - inZ)/(row['r'] - inR)) < constants.maxCtg
		outerHitSet = np.array([hit for hit in outerHitSet if filter_horizontal_doublets(hit)])
		
		'''print(f'outerHitSet ({len(outerHitSet)}) for ({innerHit}) after CUT 5: ')
		print(outerHitSet)
		triple_space()'''
		
		for outerHit in outerHitSet:
			doubletsStorage.doublets.append([innerHit['hit_id'], outerHit['hit_id']])
       
	if time_event:
		end = clock() - start
		print(f'RUNTIME: .../seeding/doublet_making.py  - {end} sec')



def debug_hit_table(table, storage):
	'''
	This function iterates through the chunks and test if the hitTable and spStorage objects
	contain the same number of hits in each chunk.
	'''
	for phiIdx in range(len(storage.phiSlices)):
		for layerIdx in range(len(storage.phiSlices[0].layerBegin)):
			hitCount = 0 
			for row in table:
				if row['phi_id'] == phiIdx and row['layer_id'] == layerIdx:
					hitCount += 1
			hitsInStorage = storage.phiSlices[phiIdx].layerEnd[layerIdx] - storage.phiSlices[phiIdx].layerBegin[layerIdx]
			print(f'Chunk ({phiIdx}, {layerIdx}) = {hitCount == hitsInStorage}')
			hitCount = 0

		

def triple_space():
	print()
	print()
	print()



