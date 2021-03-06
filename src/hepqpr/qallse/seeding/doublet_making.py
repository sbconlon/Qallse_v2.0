from .topology import DetectorModel
from hepqpr.qallse.data_wrapper import * 

from time import time
import numpy as np
from numba import jit, guvectorize, prange
from numba import int64, float32, boolean

def doublet_making(truth_path=None, hits_path=None, truth=None, hits=None, test_mode=False):
	
	
	#------ Define Constants
	time_event, debug = True, False
	nPhiSlices = 53
	nLayers = 10
	maxDoubletLength = 300.0
	minDoubletLength = 10.0
	zPlus = 150
	zMinus = -150
	maxEta = 2.7
	maxTheta = 2 * np.arctan(np.exp(-maxEta))
	maxCtg = np.cos(maxTheta) / np.sin(maxTheta) 
	detModel = DetectorModel.buildModel_TrackML()
	modelLayers = np.zeros(detModel.layers.shape)
	np.copyto(modelLayers,detModel.layers)
	FALSE_INT = 99999   #Integer that represents a false value

        #------ Load Hit Data
	truth = pd.read_csv(truth_path, index_col=False) if truth is None else truth.copy()
	hits = pd.read_csv(hits_path, index_col=False) if hits is None else hits.copy()
	hit_df = hits.copy()
	hit_df['phi_bin'] = bin_phi(hit_df['x'].values, hit_df['y'].values, nPhiSlices)
	hit_df['r'] = np.hypot(hit_df['x'].values, hit_df['y'].values)
	layer_bin = np.ones(hit_df.shape[0]) * FALSE_INT
	layNoToIdx = {2: 0, 4: 1, 6: 2, 8: 3}
	volToOffset = {8: 0, 13: 4, 17: 8}
	counter = 0
	for _, row in hit_df.iterrows():
		layer_bin[counter] = layNoToIdx[row['layer_id']] + volToOffset[row['volume_id']]
		counter += 1
	hit_df['layer_bin'] = layer_bin
	hit_df.drop(columns=['x', 'y', 'volume_id', 'module_id', 'layer_id'], inplace=True)
	cols = hit_df.columns.tolist()
	cols = [cols[0], cols[4], cols[2], cols[3], cols[1]]
	hit_df = hit_df[cols]
	hit_table = hit_df.values.astype(np.int64)
	nHits = hit_table.shape[0]
	

	#------ Start Internal Helper Functions
	
	@jit(nopython=True)
	def filter(table, inner_hit, layer_range, z_ranges):
		'''
		This function combines the helper filters into one filter and is compiled by numba into a general numpy universal function
		'''
		keep = np.array([True] * table.shape[0])
		for row_idx in range(table.shape[0]):
			keep[row_idx] = (filter_layers(table[row_idx][1], layer_range) and
		                     filter_phi(inner_hit[2], table[row_idx][2], nPhiSlices) and
		                     filter_doublet_length(inner_hit[3], table[row_idx][3], minDoubletLength, maxDoubletLength) and
		                     filter_horizontal_doublets(inner_hit[3], inner_hit[4], table[row_idx][3], table[row_idx][4], maxCtg) and
		                     filter_z(table[row_idx][1], table[row_idx][4], layer_range, z_ranges))
		return keep



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
		z_mask(layer_range, z_ranges, modelLayers, FALSE_INT)

		return layer_range, z_ranges
		
		
	@jit(nopython=True, parallel=True)
	def make():
		'''
		This function makes all possible doublets that fit the criteria of the filter. It first 
		choses an inner hit and then iterates through the hit table looking for possible outer 
		hit candidates. It then returns a list of hit ids cooresponding to the inner and outer
		hit pairs of the created doublets.
		'''
		ncolumns = int(nHits * 0.01)
		outer_2D = np.zeros((nHits, ncolumns), dtype=int64)

		for row_idx in prange(nHits):
			inner_hit = hit_table[row_idx]
			layer_range, z_ranges = get_valid_ranges(inner_hit)
			outer_hit_set = hit_table[filter(hit_table, inner_hit, layer_range, z_ranges)].T[0]
			for column_idx in prange(len(outer_hit_set)):
				outer_2D[row_idx][column_idx] = outer_hit_set[column_idx]
		
		
		outer = np.reshape(outer_2D, (1, nHits * ncolumns))[0]
		inner = np.zeros(len(outer), dtype=int64)
		for row_count in prange(outer_2D.shape[0]):
			for col_count in prange(ncolumns):	
				inner[(row_count * ncolumns + col_count)] = hit_table[row_count][0]
		
		return inner, outer
			

	#------ End Internal Helper Functions
	
		
	
	
		

	#------ Start Main Functionality
	
	hit_table.setflags(write=False)         #make hit_table immutable
	
	if debug:
		print('Hit_Table Dims: ', hit_table.shape)
		
	if time_event:
		start = time()
	
	inner_ids, outer_ids = make()
	
	doublets = pd.DataFrame({'inner': inner_ids, 'outer': outer_ids})
	doublets.drop_duplicates(inplace=True, keep=False)
				
	if time_event:
		runtime = time() - start
		if debug:
			print(f'RUNTIME: .../seeding/doublet_making.py  - {runtime} sec')
			
	if not (test_mode or debug):
		return doublets
		

	#------ End Main Functionality 
	
	
	
	
	#------ Start Debug
	
	if debug:
		print('--Grading Doublets--')
		dataw = DataWrapper(hits, truth)
		p, r, ms = dataw.compute_score(doublets)
		print(f'Precision: {p}')
		print(f'Recall: {r}')
		print(f'Missing the following {len(ms)} doublets:')
		
		for miss in ms:
			print('-------------------------------------------------')
			innerHit = hit_table[where(miss[0], hit_table.T[0])]
			outerHit = hit_table[where(miss[1], hit_table.T[0])]
			print('InnerHit: ', innerHit)
			print('OuterHit: ', outerHit)
			layer_range, z_ranges = get_valid_ranges(innerHit)
			print('filter_layers: ', filter_layers(outerHit[1], layer_range))
			print('filter_phi: ', filter_phi(outerHit[2], innerHit[2], nPhiSlices))
			print('filter_doublet_length: ', filter_doublet_length(innerHit[3], outerHit[3], minDoubletLength, maxDoubletLength))
			print('filter_horizontal_doublets: ', filter_horizontal_doublets(innerHit[3], innerHit[4], outerHit[3], outerHit[4], maxCtg))
			print('filter_z: ', filter_z(outerHit[1], outerHit[4], layer_range, z_ranges))
			print('-------------------------------------------------')
	
	if test_mode:
		dataw = DataWrapper(hits, truth)
		p, r, ms = dataw.compute_score(doublets)
		doublet_making_result = [round(runtime, 2), round(r, 2), round(p, 2), doublets.shape[0]]
		return doublet_making_result
			
	#------ End Debug






#------ Start External Helper Functions

@jit(nopython=True)
def filter_layers(layer_id, layer_range, verbose=False):
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
def filter_z(outer_layer, outer_z, layer_range, z_ranges):
	return (outer_z > z_ranges[outer_layer][0] and outer_z < z_ranges[outer_layer][1])
	
@jit(nopython=True)
def get_layer_range(inner_hit, layer_radii, nLayers, maxDoubletLength, FALSE_INT):
	'''
	This function, given a inner hit, returns a list of layers that may contain valid outer hits
	'''
	valid_layers = []
	for layer_id in range(nLayers):
		if (layer_id == inner_hit[1]+1 or layer_id == inner_hit[1]+2 or
		    layer_id == inner_hit[1]-1 or layer_id == inner_hit[1]-2):
			valid_layers.append(layer_id)
		else:
			valid_layers.append(FALSE_INT)
	return valid_layers
	
@jit(nopython=True)
def get_z_ranges(inner_hit, refCoords, layer_range, zMinus, zPlus, FALSE_INT):
	'''
	This function, given an inner hit, calculates the z region of interest for all valid layers in layer_range
	'''
	z_ranges = np.zeros((len(layer_range), 2), dtype=int64)
	for idx in range(len(layer_range)):
		if layer_range[idx] == FALSE_INT:
			z_ranges[idx][0], z_ranges[idx][1] = FALSE_INT, FALSE_INT
		else:
			z_minus = zMinus + refCoords[idx] * (inner_hit[4] - zMinus) // inner_hit[3]
			z_plus  = zPlus  + refCoords[idx] * (inner_hit[4] - zPlus) // inner_hit[3]
			z_ranges[idx][0], z_ranges[idx][1] = min(z_minus, z_plus), max(z_minus, z_plus)
	return z_ranges
	
@jit(nopython=True)
def z_mask(layer_range, z_ranges, layerModels, FALSE_INT):
	'''
	This function sets the elements in layer_range and z_ranges to FLASE_INT if their corresponding layer geometries
	are outside the z range
	'''
	for idx in range(len(layer_range)):
		if not layer_range[idx] == FALSE_INT:
			if not (layerModels[layer_range[idx]][3] > z_ranges[idx][0] and layerModels[layer_range[idx]][2] < z_ranges[idx][1]):
				layer_range[idx] = FALSE_INT
				z_ranges[idx][0] = FALSE_INT
				z_ranges[idx][1] = FALSE_INT
		
@jit(nopython=True)
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
	
@jit(nopython=True)
def zip(x, y):
	'''
	This function is the numba friendly implementation of the zip function in python 
	'''
	zipped = np.array([])
	for idx in range(len(x)):
		zipped = append(zipped, np.array([x[idx], y[idx]]))
	return zipped
	
@jit(nopython=True)
def append(arr, val):
	'''
	This function is the numba friendly implementation of the append function in numpy
	'''
	new_arr = np.zeros(len(arr) + 1, dtype=int64)
	for idx in range(len(arr)):
		new_arr[idx] = arr[idx]
	new_arr[idx+1] = val
	return new_arr
	
def bin_phi(x, y, nbins):
	phi = np.arctan2(y, x)
	for idx in range(len(phi)):
		if phi[idx] < 0:
			phi[idx] += 2*np.pi
	return (phi * float(nbins-1) / (2*np.pi)).astype(np.int)
	

def approx_doublet_length(nhits):
	a3 = (-1.073) * pow(10, -9)
	a2 = (1.616)  * pow(10, -3)
	a1 = (-8.490) * pow(10,-2)
	a0 = 9000
	return int(a3 * pow(nhits,3) + a2 * pow(nhits,2) + a1 * nhits + a0)



def triple_space():
	print()
	print()
	print()
	

#------ End External Helper Functions



