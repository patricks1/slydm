import fitting
import pickle
import paths
import h5py
import time
import scipy
import pandas as pd
import multiprocessing as mp
import numpy as np

N = 25 
maximum = 0.15

dhfracs = np.linspace(0.0, maximum, N)
Nh = len(dhfracs)

fname = 'grid.h5'

def get_df():
    df = pd.read_pickle(paths.data + 'dm_stats_20221208.pkl')
    return df

def base_op(ddfrac):
    df = get_df()
    #pool = mp.Pool(mp.cpu_count())
    #thingy = [pool.apply(fitting.count_within, args=(ddfrac, dhfrac, df, False,
    #                                                 True)) 
    #          for dhfrac in dhfracs]
    #pool.close()
    thingy = np.zeros((Nh, 4))
    for i, dhfrac in enumerate(dhfracs):
        thingy[i] = fitting.count_within_agg(ddfrac, dhfrac, df, 
                                             assume_corr=False, 
                                             return_fracs=True)
    return thingy
    
def grid_eval(ddfracs, dhfracs):
    assert len(ddfracs.shape) == 1 and len(dhfracs.shape) == 1
    Nd = len(ddfracs)
    PERCENT_WITHIN = np.zeros((Nd, Nh))
    AREA = np.zeros((Nd, Nh))

    pool = mp.Pool(mp.cpu_count())
    info = pool.map(base_op, ddfracs)
    info = np.array(info)
    pool.close()

    PERCENT_WITHIN = info[:,:,0]
    AREA_WITHIN = info[:,:,1] 
    DDFRAC = info[:,:,2]
    DHFRAC = info[:,:,3]
    
    with h5py.File(paths.data + 'grid.h5', 'w') as f:
        f.create_dataset('ddfracs', data=DDFRAC)
        f.create_dataset('dhfracs', data=DHFRAC)
        f.create_dataset('percents', data=PERCENT_WITHIN)
        f.create_dataset('areas', data=AREA_WITHIN)
    return PERCENT_WITHIN

def load_data():
    with h5py.File(paths.data + 'grid.h5', 'r') as f:
        #grid = np.array(f['grid'])
        percents = np.array(f['percents'])
        areas = np.array(f['areas'])
        dhfracs = np.array(f['dhfracs'])
        ddfracs = np.array(f['ddfracs'])
    return percents, areas, ddfracs, dhfracs

def identify():
    percents, areas, ddfracs, dhfracs = load_data()
    percents_flat = percents.flatten()
    indices_order = np.argsort(percents_flat)
    P1std = scipy.special.erf(1. / np.sqrt(2.))
    # At least 68.29% of the data contained in the band:
    over683 = percents_flat[indices_order] >= P1std 
    i_answer = np.min(np.arange(len(percents_flat))[over683])
    ddfrac = ddfracs.flatten()[indices_order][i_answer]
    dhfrac = dhfracs.flatten()[indices_order][i_answer]
    return ddfrac, dhfrac

if __name__ == '__main__':
    start = time.time()
    
    ddfracs = np.linspace(0.0, maximum, N)

    print(grid_eval(ddfracs, dhfracs))

    elapsed = time.time() - start
    minutes = elapsed // 60.
    sec = elapsed - minutes * 60.
    print('{0:0.0f}min, {1:0.1f}s taken to evaluate.'.format(minutes, sec)) 
