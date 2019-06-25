import os
import re

import click

from .config import *
from .doublet_making import doublet_making
from .storage import *
from .topology import DetectorModel
from .utils import *

from hepqpr.qallse.data_wrapper import *

def generate_doublets(truth=None, hits=None, truth_path=None, hit_path=None, config_cls=HptSeedingConfig) -> pd.DataFrame:
    det = DetectorModel.buildModel_TrackML()
    n_layers = len(det.layers)

    truth = pd.read_csv(truth_path, index_col=False) if truth is None else truth.copy()
    truth = truth.iloc[np.where(np.in1d(hits['volume_id'], [8, 13, 17]))] 
    hits = pd.read_csv(hits_path, index_col=False) if hits is None else hits.copy()
    hits = hits.iloc[np.where(np.in1d(hits['volume_id'], [8, 13, 17]))]

    config = config_cls(n_layers)
    dataw = DataWrapper(hits, truth)
    
    # constructing hit_table
    hit_table = hits[:]
    hit_table['phi'] = calc_phi(hit_table['x'], hit_table['y'])
    hit_table['phi_id'] = scale_phi(hit_table['phi'], config.nPhiSlices)
    hit_table['z_id'], z_map = scale_z(hit_table['z'], minz= hit_table['z'].min(), maxz= hit_table['z'].max(), nbins= 10000)  #adjust min, max, bin size for optimal run time and precision  
    hit_table.drop(columns=['y', 'volume_id', 'module_id', 'phi'], inplace=True)
    hit_table.set_index(['layer_id', 'phi_id', 'z_id'], inplace=True)
    
	#return the constructed doublets
    return doublet_making(config, hit_table, det, dataw, z_map)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-o', '--out', default=None)
@click.option('--score/--no-score', is_flag=True, default=True)
@click.argument('hits_path', default='/tmp/barrel_100/event000001000')
def cli(out=None, score=True, hits_path=None):
    '''
    Generate initial doublets.
    '''
    path = hits_path.replace('-hits.csv', '')
    event_id = re.search('(event[0-9]+)', hits_path)[0]
    if out is None: out = os.path.dirname(hits_path)

    print(f'Loading file {hits_path}')
    hits = pd.read_csv(path + '-hits.csv').set_index('hit_id', drop=False)

    doublets_df = generate_doublets(hits=hits)
    print(f'found {doublets_df.shape[0]} doublets.')

    if score:
        from hepqpr.qallse.data_wrapper import DataWrapper
        dw = DataWrapper.from_path(path)
        p, r, ms = dw.compute_score(doublets_df.values)
        print(f'DBLETS SCORE -- precision {p * 100}%, recall: {r * 100}% (missing doublets: {len(ms)})')

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f'{event_id}-doublets.csv'), 'w') as f:
        doublets_df.to_csv(f, index=False)
        print(f'doublets written to {f.name}')

    print('done')
