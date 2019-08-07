import numpy as np
from numba import jitclass
from numba import boolean, float32, int32

@jitclass([
			('type', boolean),
			('refCoord', float32),
			('minBound', int32),
			('maxBound', int32)])
class SiliconLayer:
    """
    Holds informations about a single layer, extracted from the detector geometry
    """

    def __init__(self, ltype, refCoord, minBound, maxBound):
        # 0 for a barrel layer, +/- 2 for positive/negative endcap layers
        self.type = ltype
        # coordinate which define the layer, from the atlas geometry --> r for barrel, z for endcap
        self.refCoord = refCoord
        # min coordinate of the layer, z for barrel, r for endcap
        self.minBound = minBound
        # max coordinate of the layer, z for barrel, r for endcap
        self.maxBound = maxBound


class DetectorModel:
    """
    Holds all the layers present in the detector.
    For now only support a detector with 4 Pixel and 4 SCT layers is implemented, without endcaps
    """

    def __init__(self):
        self.layers = None

    @staticmethod
    def buildModel_TrackML():
        """
        Build detector model type 1, only pixel and SCT layers, without endcaps.
        Geometry derived from the ACTS detector.
        """
        det = DetectorModel()
        # values from the ATLAS inner detector geometry, layerIdx 0 is the innerMost pixel layer
        # order
        # 0: ltype, 1: refCoord, 2: minBound, 3: maxBound
        det.layers = np.array([
						[0, 32, -455, 455],  # 8-2
						[0, 72, -455, 455],  # 8-4
						[0, 116, -455, 455],  # 8-6
						[0, 172, -455, 455],  # 8-8
						[0, 260, -1030, 1030],  # 13-2
						[0, 360, -1030, 1030],  # 13-4
						[0, 500, -1030, 1030],  # 13-6
						[0, 660, -1030, 1030],  # 13-8
						[0, 820, -1030, 1030],  # 17-2
						[0, 1020, -1030, 1030]  # 17-4
        ], dtype='int32')
        return det
