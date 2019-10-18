import os
import re

import click

from .config import *
from .doublet_making import doublet_making
from .storage import *
from .topology import DetectorModel
from .utils import *

from hepqpr.qallse.data_wrapper import *


def generate_doublets(*args, **kwargs):
    if 'test_mode' in kwargs and kwargs['test_mode']:
        return run_seeding(*args, **kwargs)
    else:
        seeding_results = run_seeding(*args, **kwargs)
        doublets = structures_to_doublets(*seeding_results)
        doublets_df = pd.DataFrame(doublets, columns=['start', 'end'])
        return doublets_df


def run_seeding(truth_path=None, hits_path=None, truth=None, hits=None, config_cls=HptSeedingConfig, test_mode=False):
    det = DetectorModel.buildModel_TrackML()
    n_layers = len(det.layers)

    truth = pd.read_csv(truth_path, index_col=False) if truth is None else truth.copy()
    truth = truth.iloc[np.where(np.in1d(hits['volume_id'], [8, 13, 17]))] 
    hits = pd.read_csv(hits_path, index_col=False) if hits is None else hits.copy()
    hits = hits.iloc[np.where(np.in1d(hits['volume_id'], [8, 13, 17]))]

    config = config_cls(n_layers)
    dataw = DataWrapper(hits, truth)
    spStorage = SpacepointStorage(hits, config)
    doubletsStorage = DoubletStorage()
    
    if test_mode:
        return doublet_making(config, spStorage, det, doubletsStorage, dataw, test_mode=True)
		
    doublet_making(config, spStorage, det, doubletsStorage, dataw)

    # returning the results
    return hits, spStorage, doubletsStorage


def structures_to_doublets(hits: pd.DataFrame = None, sps: SpacepointStorage = None, ds: DoubletStorage = None):
    return pd.DataFrame({'inner': ds.inner, 'outer': ds.outer})


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
