from . import fitting
import pickle
import paths
import h5py
import time
import scipy
import sys
import pandas as pd
import multiprocessing as mp
import numpy as np

N = 25 

ddfrac_min = 0.2
ddfrac_max = 0.35

dpfrac_min = 0.2
dpfrac_max = 0.35

dpfracs = np.linspace(dpfrac_min, dpfrac_max, N)

with open(paths.data + 'results_mao_lim_fit.pkl', 'rb') as f:
    params = pickle.load(f)

# Location to store / from which to pull grid data
def get_df():
    # It seems to break with h5py but not with pickle. I can't remember why.
    df = pd.read_pickle(paths.data + 'dm_stats_dz1.0_20230724.pkl')
    return df

def base_op(ddfrac):
    df = get_df()
    #pool = mp.Pool(mp.cpu_count())
    #thingy = [pool.apply(fitting.count_within, args=(ddfrac, dhfrac, df, False,
    #                                                 True)) 
    #          for dhfrac in dhfracs]
    #pool.close()
    thingy = np.zeros((N, 4))
    for i, dpfrac in enumerate(dpfracs):
        thingy[i] = fitting.count_within_agg_mao(ddfrac, dpfrac, df, params)
    return thingy
    
def grid_eval(ddfracs, fname):
    Nd = len(ddfracs)
    PERCENT_WITHIN = np.zeros((Nd, 1))
    AREA = np.zeros((Nd, 1))

    pool = mp.Pool(mp.cpu_count())
    info = pool.map(base_op, ddfracs)
    info = np.array(info)
    pool.close()

    PERCENT_WITHIN = info[:,:,0]
    AREA_WITHIN = info[:,:,1] 
    DDFRAC = info[:,:,2]
    DPFRAC = info[:, :, 3]
    
    with h5py.File(paths.data + fname, 'w') as f:
        f.create_dataset('ddfracs', data=DDFRAC)
        f.create_dataset('dpfracs', data=DPFRAC)
        f.create_dataset('percents', data=PERCENT_WITHIN)
        f.create_dataset('areas', data=AREA_WITHIN)
    return PERCENT_WITHIN

def load_data(fname):
    with h5py.File(paths.data + fname, 'r') as f:
        #grid = np.array(f['grid'])
        percents = np.array(f['percents'])
        areas = np.array(f['areas'])
        ddfracs = np.array(f['ddfracs'])
        dpfracs = np.array(f['dpfracs'])
    return percents, areas, ddfracs, dpfracs

def identify(fname):
    percents, areas, ddfracs, dpfracs = load_data(fname)
    percents_flat = percents.flatten()
    indices_order = np.argsort(percents_flat)
    P1std = scipy.special.erf(1. / np.sqrt(2.))
    # At least 68.29% of the data contained in the band:
    over683 = percents_flat[indices_order] >= P1std 
    i_answer = np.min(np.arange(len(percents_flat))[over683])
    ddfrac = ddfracs.flatten()[indices_order][i_answer]
    dpfrac = dpfracs.flatten()[indices_order][i_answer]
    return ddfrac, dpfrac

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
        fname = 'grid_mao.h5' 
    else:
        fname = args[1]

    ddfracs = np.linspace(ddfrac_min, ddfrac_max, N)

    print(grid_eval(ddfracs, fname))

    elapsed = time.time() - start
    minutes = elapsed // 60.
    sec = elapsed - minutes * 60.
    print('{0:0.0f}min, {1:0.1f}s taken to evaluate.'.format(minutes, sec)) 
