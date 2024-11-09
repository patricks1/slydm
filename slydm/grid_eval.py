import fitting
import pickle
import paths
import h5py
import time
import scipy
import sys
import pandas as pd
import multiprocessing as mp
import numpy as np

N = 26 
maximum = 0.16

dhfracs = np.linspace(0.0, maximum, N)
Nh = len(dhfracs)


# Specify whether to override fit results with results from fitting on the 
# halo integral
override = False 
if override:
    fname = 'grid_gfit.h5' # Location to store / from which to pull grid data
    with open(paths.data + 'data_raw_gfit.pkl', 'rb') as f: 
        # Give fitting.count_within_agg hard-coded parameters from fitting
        # for the halo integral
        data_override = pickle.load(f)
else:
    # Location to store / from which to pull grid data
    fname = 'grid_20230713.h5' 
    # Don't give fitting.count_within_agg hard-coded parameters. Let it look
    # up the parameters stored in data_raw.pkl.
    data_override = None

def get_df():
    # It seems to break with h5py but not with pickle. I can't remember why.
    df = pd.read_pickle(paths.data + 'dm_stats_dz1.0_20230626.pkl')
    #import dm_den
    #df = dm_den.load_data('dm_stats_dz1.0_20230626.h5')
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
                                             return_fracs=True, 
                                             data_override=data_override)
    return thingy
    
def grid_eval(ddfracs, dhfracs, fname=fname):
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
    
    with h5py.File(paths.data + fname, 'w') as f:
        f.create_dataset('ddfracs', data=DDFRAC)
        f.create_dataset('dhfracs', data=DHFRAC)
        f.create_dataset('percents', data=PERCENT_WITHIN)
        f.create_dataset('areas', data=AREA_WITHIN)
    return PERCENT_WITHIN

def load_data():
    with h5py.File(paths.data + fname, 'r') as f:
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
    '''
    User can provide a system argument.
    argv[0] is just the script name.
    argv[1] should be the fname to which the script saves the grid.
    '''
    start = time.time()
   
    args = sys.argv
    if len(args) < 2:
        # User did not specify a file name.
        fname = fname
    else:
        fname = args[1]

    ddfracs = np.linspace(0.0, maximum, N)

    print(grid_eval(ddfracs, dhfracs, fname))

    elapsed = time.time() - start
    minutes = elapsed // 60.
    sec = elapsed - minutes * 60.
    print('{0:0.0f}min, {1:0.1f}s taken to evaluate.'.format(minutes, sec)) 
