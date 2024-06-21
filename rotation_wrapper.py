from UCI_tools import rotate_galaxy
import dm_den
import pickle
import os
import numpy as np
import pandas as pd
from astropy import units as u, constants as c
from UCI_tools import staudt_tools
from UCI_tools import tools as uci

path = '/DFS-L/DATA/cosmo/grenache/staudt/dm_den/'

def get_rotated_gal(df, galname):
    ''' 
    Returns a dictionary of the given galaxy including rotated vectors, where
    the galaxy is rotated based on all three of stars, DM, and gas.
    '''

    #dark matter and stars
    dm_stars = dm_den.unpack_new(df, galname,
                                 getparts=['PartType1','PartType4'])
    gas = dm_den.unpack_gas(df, galname)

    #gas dictionary
    gas_dict = dict([(name,data) \
                     for name, data in zip(['mass','r','v_mag',
                                            'v_vec',
                                            'parttype',
                                            'coord','energy',
                                            'e_abundance',
                                            'he_frac'],
                                           [gas[i] for i in \
                                            [0,2]+list(range(4,11))])])
    #dark matter and stars dictionary
    dms_dict = dict([(name,data) \
                     for name, data in zip(['mass','r','v_mag',
                                            'v_vec',
                                            'parttype',
                                            'coord'],
                                           [gas[i] for i in \
                                            [0,2]+list(range(4,8))])])
    cen_coord = dm_stars[8]

    d = {}
    # Take the outputs of dm_den.unpack_new and dm_den.unpack_gas, which are
    # just np.ndarray's, combine gas, dm, and stars, and put them
    # into a dictionary 
    for name, i in zip(['mass','r','v_mag','v_vec','parttype','coord'],
                       [0,2,4,5,6,7]):
        d[name] = np.append(gas[i], dm_stars[i], axis=0)

    d['coord_centered'] = d['coord']-cen_coord

    for name, i in zip(['energy','e_abundance','he_frac'],
                       [8,9,10]):
        d[name] = np.append(gas[i], np.repeat(np.nan,len(dm_stars[0])))

    l = len(d['mass'])
    for key in d.keys():
        try:
            assert len(d[key])==l
        except:
            print('failed on '+key)

    isgas = d['parttype']=='PartType0'
    Ts = uci.calc_temps(*[d[data][isgas] for data in ['he_frac',
                                                         'e_abundance',
                                                         'energy']])
    d['T'] = np.append(Ts, np.repeat(np.nan,len(dm_stars[0])))

    # Rotate the galaxy based on stars, stars, and DM (`d` contains all 3 of
    # those.)
    dat_rot = rotate_galaxy.rotate_gal(*[d[data] for data in ['coord_centered',
                                                              'v_vec',
                                                              'mass',
                                                              'r']])
    d['coord_rot'] = dat_rot[0] #rotated coordinates
    d['v_vec_rot'] = dat_rot[1] #rotated velocity vector

    d['v_vec_disc'] = d['v_vec_rot'][:,:2] #xy component of velocity
    d['coord_disc'] = d['coord_rot'][:,:2] #xy component of coordinates

    # Find the projection of velocity onto the xy vector (i.e. v_r)
    # dot(V, R) / R^2 * R = V_{||R} 
    # Note, the projection of V along R is not the same as the x,y
    #     components of V. In fact, the magnitude of V along R <= the
    #     magnitude of V.
    d['v_r_vec'] = (np.sum(d['v_vec_disc'] * d['coord_disc'], axis=1) \
                   / np.linalg.norm(d['coord_disc'], axis=1) **2.) \
                   .reshape(len(d['coord_disc']), -1) \
                   * d['coord_disc']

    #v_phi is the difference of the xy velocity and v_r
    d['v_phi_vec'] = d['v_vec_disc'] - d['v_r_vec']
    d['v_phi_mag'] = np.linalg.norm(d['v_phi_vec'],axis=1)

    l = len(d['mass'])
    for key in d.keys():
        try:
            assert len(d[key])==l
        except:
            print('failed on '+key)

    return d

def calc_vphimagcool(df, galname):
    ''' Return the magnitudes of the galaxy's v_phi vectors '''

    d = get_rotated_gal(df, galname)

    r0 = 8.3
    dr = 1.5
    within_shell = (d['r']>=r0-dr/2.) & (d['r']<=r0+dr/2.) #from full dataset

    l = len(d['mass'])
    for key in d.keys():
        try:
            assert len(d[key])==l
        except:
            print('failed on '+key)

    iscool = d['T'].to(u.K).value<10.**(3.) #from gas dataset
    vphi_mags_cool = d['v_phi_mag'][isgas & within_shell & iscool]

    return vphi_mags_cool

def gen_vphimagcool_data(fname):
    d = {}
    df = staudt_tools.init_df()
    for galname in df.index:
        print('Processing '+galname)
        d[galname]=calc_vphimagcool(df, galname)
        with open(path+fname,'wb') as f:
            pickle.dump(d, f)
    return d
