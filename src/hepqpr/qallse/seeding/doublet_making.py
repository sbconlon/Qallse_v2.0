import numpy as np
from .storage import *

from hepqpr.qallse.data_wrapper import * 
from time import clock

from numba import jit, guvectorize
from numba import int64, float32, boolean

def doublet_making(constants, spStorage: SpacepointStorage, detModel, doubletsStorage: DoubletStorage, dataw: DataWrapper):
	
	#____________________________________________#
	#                                            #
	#------------- Define Constants -------------#
	#____________________________________________#
	
	time_event, debug = True, True
	nHits = spStorage.x.size
	nPhiSlices = len(spStorage.phiSlices)
	nLayers = len(spStorage.phiSlices[0].layerBegin)
	modelLayers = np.zeros(detModel.layers.shape)
	np.copyto(modelLayers, detModel.layers)				 
	maxDoubletLength = constants.maxDoubletLength
	minDoubletLength = constants.minDoubletLength
	zPlus = constants.zPlus
	zMinus = constants.zMinus
	maxCtg = constants.maxCtg
	FALSE_INT = 99999    #Integer that reperesents a false value
	
	
	
	
	
	
	#____________________________________________#
	#                                            #
	#------ Start Internal Helper Functions -----#
	#____________________________________________#
	
	def generate_hit_table() -> np.array:
		"""
		This function transfers the information stored in spacepoint storage into a structured numpy array 
		containing the information necessary for doublet making.
		"""
		table = np.array([spStorage.hit_id,  # 0: hit id
		                  np.zeros(nHits),   # 1: layer id
		                  np.zeros(nHits),   # 2: phi id
		                  spStorage.r,       # 3: r
		                  spStorage.z],      # 4: z
		                  dtype=np.int64)      

		for phiIdx in range(nPhiSlices):
			for layerIdx in range(nLayers):
				startIdx = spStorage.phiSlices[phiIdx].layerBegin[layerIdx]
				endIdx = spStorage.phiSlices[phiIdx].layerEnd[layerIdx]
				for idx in range(startIdx, endIdx):
					table[1][idx] = layerIdx
					table[2][idx] = phiIdx
		return table.T
	
	
	
	@guvectorize([(int64[:, :], int64[:], int64[:], int64[:], int64[:], boolean[:])], "(n, m),(m),(l),(l),(o)->(n)", nopython=True)
	def filter(table, inner_hit, layers, minInterest, maxInterest, mask):
		'''
		This function combines the helper filters into one filter and is compiled by numba into a general numpy universal function
		'''
		for row_idx in range(table.shape[0]):
			#filter_layers
			if not filter_layers(table[row_idx][1], layers):
				mask[row_idx] = False

			#filter_phi
			elif not filter_phi(inner_hit[2], table[row_idx][2], nPhiSlices):
				mask[row_idx] = False
				
			#filter_doublet_length
			elif not filter_doublet_length(inner_hit[3], table[row_idx][3], minDoubletLength, maxDoubletLength):
				mask[row_idx] = False
				
			#filter_horizontal_doublets
			elif not filter_horizontal_doublets(inner_hit[3], inner_hit[4], table[row_idx][3], table[row_idx][4], maxCtg):
				mask[row_idx] = False
			
			#filter_boring_hits
			elif not filter_boring_hits(table[row_idx][1], table[row_idx][4], layers, minInterest, maxInterest):
				mask[row_idx] = False
			
			else:
				mask[row_idx] = True



	@jit(nopython=True)
	def get_valid_ranges(inner_hit):
		'''
		This function returns the list of layers that contain interesting hits, given our chosen inner hit. 
		It also returns the min/max bound in the z-direction for interesting hits for each outer layer.
		'''
		#Get the radius of each layer
		refCoords = np.array([modelLayers[layer_idx][1] for layer_idx in range(nLayers)], dtype=int64)

		#Get the list of all valid layers
		layer_range = get_layer_range(inner_hit, refCoords, nLayers, maxDoubletLength, FALSE_INT)

		#Find the z bounds for each valid layer
		z_ranges = get_z_ranges(inner_hit, refCoords, layer_range, zMinus, zPlus, FALSE_INT) 

		#Filter layers whose bounds of interest fall outside their geometric bounds 
		z_mask()

		return layer_range, maxInterest, minInterest
		
		
		
	def make():
		'''
		This function makes all possible doublets that fit the criteria of the filter. It first choses an inner hit and then iterates
		through the hit table looking for possible outer hit candidates. It chooses two inner hits in an attempt to help balance the 
		computation time for each loop.
		'''
		for indx in range(nHits//2):
			# If there are an even number of rows, the final iteration should be ignored
			if indx > (nHits-indx-1):
				continue
			# If there are an odd number of rows, then the final iteration should only use one row
			if indx == (nHits-indx-1):
				inner_hit = hit_table[indx]
				layers, maxInterest, minInterest = get_valid_ranges(inner_hit)
				outerHitSet_one = np.array(filter(hit_table, inner_hit, layers_one, minInterest_one, maxInterest_one))
				for outerHit in outerHitSet_one:
					doubletsStorage.inner.append(hit_table[indx][0])
					doubletsStorage.outer.append(outerHit[0])
			# Otherwise, two rows should be used
			else:
				inner_hit_one = hit_table[indx]
				inner_hit_two = hit_table[nHits-indx-1]
				layers_one, maxInterest_one, minInterest_one = get_valid_ranges(inner_hit_one)
				layers_two, maxInterest_two, minInterest_two = get_valid_ranges(inner_hit_two)
				outerHitSet_one = hit_table[filter(hit_table, inner_hit_one, layers_one, minInterest_one, maxInterest_two)]
				outerHitSet_two = hit_table[filter(hit_table, inner_hit_two, layers_two, minInterest_two, maxInterest_two)]
				for outer_hit_one in outerHitSet_one:
					doubletsStorage.inner.append(inner_hit_one[0])
					doubletsStorage.outer.append(outer_hit_one[0])
				for outer_hit_two in outerHitSet_two:
					doubletsStorage.inner.append(inner_hit_two[0])
					doubletsStorage.outer.append(outer_hit_two[0])
				
		print(f'---> {len(doubletsStorage.inner)} Doublets Created')
			
	#____________________________________________#
	#                                            #
	#------- End Internal Helper Functions ------#
	#____________________________________________#
	
		
	
	
		
	#____________________________________________#		
	#                                            #
	#----------Start Main Functionality----------#
	#____________________________________________#
	
	hit_table = generate_hit_table()
	hit_table.setflags(write=False)         #make hit_table immutable
	
	if debug:
		print('Hit_Table Dims: ', hit_table.shape)
		debug_hit_table(hit_table, spStorage)
		
	if time_event:
		start = clock()
	
	make()
				
	if time_event:
		end = clock() - start
		print(f'RUNTIME: .../seeding/doublet_making.py  - {end} sec')
		
	#____________________________________________#
	#                                            #
	#---------- End Main Functionality ----------#
	#____________________________________________#
	
	
	
			
			
	#____________________________________________#
	#                                            #
	#--------------- Start Debug ----------------#
	#____________________________________________#
	
	if debug:
		doublets = pd.DataFrame({'inner': doubletsStorage.inner, 'outer': doubletsStorage.outer})
		print('--Grading Doublets--')
		p, r, ms = dataw.compute_score(doublets)
		print(f'Purity: {p * 100}% real doublets')
		print(f'Missing the following {len(ms)} doublets:')
		for miss in ms:
			print('-------------------------------------------------')
			innerHit = hit_table[where(miss[0], hit_table.T[0])]
			outerHit = hit_table[where(miss[1], hit_table.T[0])]
			print('InnerHit: ', innerHit)
			print('OuterHit: ', outerHit)
			layers, minInterest, maxInterest = get_valid_ranges(innerHit)
			print('filter_layers: ', filter_layers(outerHit[1], layers))
			print('filter_phi: ', filter_phi(outerHit[2], innerHit[2], nPhiSlices))
			print('filter_doublet_length: ', filter_doublet_length(innerHit[3], outerHit[3], minDoubletLength, maxDoubletLength))
			print('filter_horizontal_doublets: ', filter_horizontal_doublets(innerHit[3], innerHit[4], outerHit[3], outerHit[4], maxCtg))
			print('filter_boring_hits: ', filter_boring_hits(outerHit[1], outerHit[4], layers, minInterest, maxInterest, verbose=True))
			print('-------------------------------------------------')
			
	#____________________________________________#
	#                                            #
	#----------------- End Debug ----------------#
	#____________________________________________#





#____________________________________________#
#                                            #
#------ Start External Helper Functions -----#
#____________________________________________#

@jit(nopython=True)
def filter_layers(layer_id, layer_range):
	return contains(layer_id, layer_range)
		
@jit(nopython=True)
def filter_phi(outer_phi, inner_phi, nPhiSlices):
	return ((outer_phi - 1) == inner_phi or 
		    (outer_phi + 1) == inner_phi or 
			 outer_phi == inner_phi or
		    (outer_phi == 0 and inner_phi == nPhiSlices - 2) or
		    (outer_phi == nPhiSlices - 2 and inner_phi == 0))
			    
@jit(nopython=True)
def filter_doublet_length(inner_r, outer_r, minDoubletLength, maxDoubletLength):
	return (((outer_r - inner_r) < maxDoubletLength) and ((outer_r - inner_r) > minDoubletLength))
		
@jit(nopython=True)
def filter_horizontal_doublets(inner_r, inner_z, outer_r, outer_z, maxCtg):
	return np.abs((outer_z - inner_z)/(outer_r - inner_r)) < maxCtg
		
@jit(nopython=True)
def filter_boring_hits(outer_layer, outer_z, layer_range, min_z_range, max_z_range, verbose=False):
	if verbose:
		print('Z-Min: ', min_z_range[where(outer_layer, layer_range)], '  Z-Max: ',  max_z_range[where(outer_layer, layer_range)])
		print('outer-z: ', outer_z)
	return (outer_z > min_z_range[where(outer_layer, layer_range)] and outer_z > max_z_range[where(outer_layer, layer_range)])
	
@jit(nopython=True)
def get_layer_range(inner_hit, layer_radii, nLayers, maxDoubletLength, FALSE_INT):
	'''
	This function, given a inner hit, returns a list of layers that may contain valid outer hits
	'''
	valid_layers = []
	for layer_id in range(nLayers):
		if np.abs(layer_radii[layer_id] - inner_hit[3]) < maxDoubletLength:
			valid_layers.append(layer_id)
		else:
			valid_layers.append(FALSE_INT)
	return valid_layers
	
@jit(nopython=True)
def get_z_ranges(inner_hit, refCoords, layer_range, zMinus, zPlus, FALSE_INT):
	'''
	This function, given an inner hit, calculates the z region of interest for all valid layers in layer_range
	'''
	z_min, z_max = [], []
	for idx in range(len(layer_range)):
		if layer_range[idx] == FALSE_INT:
			z_min.append(FALSE_INT)
			z_max.append(FALSE_INT)
		else:
			z_minus = zMinus + refCoords[idx] * (inner_hit[4] - zMinus) // inner_hit[3]
			z_plus  = zPlus  + refCoords[idx] * (inner_hit[4] - zPlus) // inner_hit[3]
			z_min.append(min(z_minus, z_plus))
			z_max.append(max(z_minus, z_plus))
	return zip(z_min, z_max)
	
@jit(nopython=True)
def z_mask(layer_range, z_ranges, layerModels, FALSE_INT)
	'''
	This function sets all layers in layer_range and z_ranges to FALSE_INT if the layer's geometric bounds fall outside
	the z range for that layer
	'''
	


[(modelLayers[layer_range[idx]][3] > minInterest[idx] and modelLayers[layer_range[idx]][2] < maxInterest[idx]) for idx in range(len(layer_range))]
idx_count = 0
for bool_idx in mask:
	if not bool_idx:
		layer_range[idx_count] = FALSE_INT
		maxInterest[idx_count] = FALSE_INT
		minInterest[idx_count] = FALSE_INT
idx_count += 1
		
@jit(boolean(int64, int64[:]), nopython=True)
def contains(val: int64, lst: int64[:]):
	'''
	This function is the numba friendly implementation of the contains built in funtion in python
	'''
	for elem in lst:
		if val == elem:
			return True
	return False
	
@jit(nopython=True)
def where(val, lst):
	'''
	This function is the numba friendly implementation of the where function in numpy
	'''
	index = 0
	for elem in lst:
		if val == elem:
			return index
		index += 1
	return index #will cause out of range error

def debug_hit_table(table, storage):
	'''
	This function iterates through the chunks and tests if the hitTable and spStorage objects
	contain the same number of hits in each chunk.
	'''
	result = []
	for phiIdx in range(len(storage.phiSlices)):
		for layerIdx in range(len(storage.phiSlices[0].layerBegin)):
			hitCount = 0 
			for row in table:
				if row[2] == phiIdx and row[1] == layerIdx:
					hitCount += 1
			hitsInStorage = storage.phiSlices[phiIdx].layerEnd[layerIdx] - storage.phiSlices[phiIdx].layerBegin[layerIdx]
			result.append(hitCount == hitsInStorage)
			hitCount = 0
	print('Hit Table Debug: ', all(result))
	
def triple_space():
	print()
	print()
	print()
	
#____________________________________________#
#                                            #
#------- End External Helper Functions ------#
#____________________________________________#



