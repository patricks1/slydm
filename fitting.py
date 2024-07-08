import scipy
import warnings
import lmfit
import pickle
import math
import paths
import staudt_utils
import copy
import time
import grid_eval
import grid_eval_mao
import h5py
import itertools
import numpy as np
import pandas as pd
import multiprocessing as mp
import UCI_tools.tools as uci
from progressbar import ProgressBar
from IPython.display import display, Latex

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['font.family'] = 'serif' 
rcParams['axes.titlesize']=24
rcParams['axes.titlepad']=15
rcParams['legend.frameon'] = True
rcParams['figure.facecolor'] = (1., 1., 1., 1.) #white with alpha=1.

with open('./data/v_pdfs_disc_dz1.0.pkl','rb') as f:
    pdfs_v=pickle.load(f)
with open('./data/vescs_rot_20230514.pkl', 'rb') as f:
    vesc_dict = pickle.load(f)
with open('./data/v_pdfs_incl_ve_20220205.pkl','rb') as f:
    pdfs_v_incl_vearth=pickle.load(f)
#with open(paths.data + 'vcut_hat_dict.pkl', 'rb') as f:
#    vcut_hat_dict = pickle.load(f)

for gal in pdfs_v:
    bins = pdfs_v[gal]['bins']
    vs = (bins[1:] + bins[:-1]) / 2.
    pdfs_v[gal]['vs'] = vs
    # Getting rid of the variable so it doesn't accidentally get used later:
    del vs 

max_bins_est = 1.e-3
min_bins_est = -1.e-3

###############################################################################
# Speed distributions
###############################################################################

def pN_mao(v, v0, vesc, p):
    '''
    Probability density before normalizing by N
    '''
    fN = np.exp(-np.abs(v) / v0) * (vesc**2. - v**2.) ** p
    pN = fN * 4. * np.pi * v**2.
    if isinstance(v, np.ndarray):
        isesc = v >= vesc
        pN[isesc] = 0.
    else:
        if v >= vesc:
            return 0
    return pN

def mao(v, v0, vesc, p):
    import collections
    v0_is_arraylike = isinstance(v0, collections.abc.Iterable)
    vesc_is_arraylike = isinstance(vesc, collections.abc.Iterable)
    v_is_arraylike = isinstance(v, collections.abc.Iterable)
    if (
            (v0_is_arraylike or vesc_is_arraylike) 
            and (not v_is_arraylike or len(v) != len(v0))
    ):
        raise ValueError('If v0 and vesc are arrays, v should be an array of'
                         ' the same'
                         ' size.')
    if not v0_is_arraylike:
        v0 = np.array([v0])
    if not vesc_is_arraylike:
        vesc = np.array([vesc])
    if type(v) == list:
        v = np.array(v)
    if type(v0) == list:
        v0 = np.array(v0)
    if type(vesc) == list:
        vesc = np.array(vesc)
    if len(v0) != len(vesc):
        raise ValueError('v0 and vesc should be the same length.')
    N_vcs = len(v0) # Implied number of circular speeds
    v0_set, vesc_set = np.unique([v0, vesc], axis=1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 
                                category=scipy.integrate.IntegrationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        N_dict = {
            (v0_, vesc_): normalize(pN_mao, (v0_, vesc_, p))
            for v0_, vesc_ in zip(v0_set, vesc_set)
        }
        Ns = np.array([N_dict[v0_, vesc_] for v0_, vesc_ in zip(v0, vesc)])
        prob_den = pN_mao(v, v0, vesc, p) / Ns
    if not v_is_arraylike and prob_den.shape == (1,):
        # Best practice is probably for the shape/type of `prob_den` to match
        # that of the `v` that the user provides. Therefore, if the user
        # provides `v` as a single numeral, I'll return a single numeral,
        # as opposed to an iterable of shape (1,), which I was previously
        # doing and which caused big problems when fitting methods were
        # run as list comprehensions.
        # 
        # It was a problem because the fit's
        # target vector `y` had shape (N_data,) while, when run in a list
        # comprehension, the fit `ys_hat` ended up with shape (N_data, 1). What
        # happens in this case is numpy broadcasts both vectors to 
        # (N_data, N_data) when performing `(ys_hat - y) ** 2.`. Then the SSE
        # is a sum of N_data x N_data numbers instead of just N_data.
        #
        # As a side note, running a fitting method
        # as a list comprehension is inefficient and unnecessary now that
        # I've implemented the `N_dict`. However, it's better to make it
        # so that, if done that way, the fitting results are not wrong.
        prob_den = prob_den[0]
    return prob_den

def pN_smooth_step_max(v, v0, vdamp, k):
    '''
    Probability density before normalizing by N
    '''
    fN = np.exp( - v**2. / v0**2. )
    pN = fN * 4. * np.pi * v**2.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 
                                category=scipy.integrate.IntegrationWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        trunc = 1. / (1. + np.exp(-k * (vdamp-v)))
    pN *= trunc
    return pN

def smooth_step_max(v, v0, vdamp, k, speedy=False):
    '''
    Sigmoid-damped Maxwellian, as opposed to the immediate cutoff
    of a Heaviside function used in trunc_max
    
    k is the strength of the sigmoid damping
    '''
    import collections

    v0_is_arraylike = isinstance(v0, collections.abc.Iterable)
    vdamp_is_arraylike = isinstance(vdamp, collections.abc.Iterable)
    v_is_arraylike = isinstance(v, collections.abc.Iterable)
    if ((v0_is_arraylike or vdamp_is_arraylike) 
        and (not v_is_arraylike or len(v) != len(v0))):
        raise ValueError('If v0 and vdamp are arrays, v should be an array of'
                         ' the same'
                         ' size.')
    if not v0_is_arraylike:
        v0 = np.array([v0])
    if not vdamp_is_arraylike:
        vdamp = np.array([vdamp])
    if type(v) == list:
        v = np.array(v)
    if type(v0) == list:
        v0 = np.array(v0)
    if type(vdamp) == list:
        vdamp = np.array(vdamp)
    if len(v0) != len(vdamp):
        raise ValueError('v0 and vdamp should be the same length.')
    N_vcs = len(v0) # Implied number of circular speeds

    v0_set, vdamp_set = np.unique([v0, vdamp], axis=1)

    if speedy:
        min_S = 1.e-4 # minimum sigmoid value
        max_v_set = vdamp_set + k * (1. / min_S - 1.)
    else:
        max_v_set = np.repeat(np.inf, N_vcs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        N_dict = {
                (v0_, vdamp_): normalize(pN_smooth_step_max, (v0_, vdamp_, k)) 
                for v0_, vdamp_ in zip(v0_set, vdamp_set)
        }
        Ns = np.array([N_dict[v0_, vdamp_] for v0_, vdamp_ in zip(v0, vdamp)])
        p = pN_smooth_step_max(v, v0, vdamp, k) / Ns
    
    if len(p) == 1:
        return p[0]
    else:
        return np.array(p)

def exp_max(v, v0, vesc):
    '''
    Maxwellian with an exponential decline (from Macabe 2010 and Lacroix et al. 
    2020)
    '''
    fN = np.exp( - v**2. / v0**2. ) - np.exp( - vesc**2. / v0**2. )
    pN = fN * 4. * np.pi * v**2.
    N1 = -2./3. * np.pi*vesc * np.exp( - vesc**2. / v0**2. )
    N2 = (3.*v0**2. + 2.*vesc**2.)
    N3 = np.pi**(3./2.) * v0**3. * scipy.special.erf(vesc/v0)
    N = N1 * N2 + N3
    p = pN / N
    
    if isinstance(v,(list,np.ndarray)):
        isesc = v>=vesc
        p[isesc] = 0.
    else:
        if v>=vesc:
            return 0.
        
    return p

def pN_max_double_hard(v, v0, vdamp, k, vesc):
    '''
    Probability density before normalizing by N for a double-truncated 
    Maxwellian. The first truncation is applied with a sigmoid of strength k
    around vdamp. 
    The second is an immediate truncation 
    at vesc.
    '''
    fN = np.exp( - v**2. / v0**2. )
    pN = fN * 4. * np.pi * v**2.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 
                                category=scipy.integrate.IntegrationWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        trunc = 1. / (1. + np.exp(-k * (vdamp-v)))
    pN *= trunc
    if isinstance(v, np.ndarray):
        isesc = v >= vesc
        pN[isesc] = 0.
    else:
        if v >= vesc:
            return 0
    return pN

def max_double_hard(v, v0, vdamp, k, vesc):
    '''
    Normalized probability density for a double-truncated 
    Maxwellian. The first truncation is applied with a sigmoid of strength k
    around vdamp. 
    The second is an immediate truncation 
    at vesc.
    '''
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        N = scipy.integrate.quad(pN_max_double_hard, 0., np.inf, 
                                 (v0, vdamp, k, vesc), epsabs=0)[0]
        p = pN_max_double_hard(v, v0, vdamp, k, vesc) / N
    return p

def pN_max_double_exp(v, v0, vdamp, k, vesc):
    '''
    Probability density before normalizing by N for a double-truncated 
    Maxwellian. The first truncation is applied with a sigmoid of strength k
    around vdamp. 
    The second is an exponential truncation that causes the function to reach
    0 at vesc.
    '''
    fN = np.exp( - v**2. / v0**2. ) - np.exp( - vesc**2. / v0**2.)
    pN = fN * 4. * np.pi * v**2.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 
                                category=scipy.integrate.IntegrationWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        trunc = 1. / (1. + np.exp(-k * (vdamp-v)))
    pN *= trunc

    if isinstance(v, (list, np.ndarray)):
        isesc = v>=vesc
        pN[isesc] = 0.
    else:
        if v>=vesc:
            return 0.

    return pN

def max_double_exp(v, v0, vdamp, k, vesc):
    '''
    Normalized probability density for a double-truncated 
    Maxwellian. The first truncation is applied with a sigmoid of strength k
    around vdamp. 
    The second is an exponential truncation that causes the function to reach
    0 at vesc.
    '''
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        N = scipy.integrate.quad(pN_max_double_exp, 0., np.inf, 
                                 (v0, vdamp, k, vesc), epsabs=0)[0]
        p = pN_max_double_exp(v, v0, vdamp, k, vesc) / N
    return p

###############################################################################
# Halo integrals
###############################################################################

def numeric_halo_integral(vbins, ps):
    v_widths = vbins[1:] - vbins[:-1]
    vs = (vbins[1:] + vbins[:-1]) / 2.
    integrand = ps / vs
    gs = [np.sum(integrand[i:] * v_widths[i:]) for i in range(len(integrand))]
    gs = np.array(gs)
    return gs

def g_smooth_step_max(vmins, v0, vesc, k):
    def integrand(v):
        pN = pN_smooth_step_max(v, v0, vesc, k) 
        return pN / v
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 
                                category=scipy.integrate.IntegrationWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        N = scipy.integrate.quad(pN_smooth_step_max, 0., np.inf,
                                 (v0, vesc, k), epsabs=0)[0]
        gN = [scipy.integrate.quad(integrand, vmin, np.inf)[0]
              for vmin in vmins]
        gN = np.array(gN)
    return gN / N

def calc_g_general(vmins, pN_func, args):
    def integrand(v):
        pN = pN_func(v, *args) 
        return pN / v
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 
                                category=scipy.integrate.IntegrationWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        N = scipy.integrate.quad(pN_func, 0., np.inf,
                                 args, epsabs=0)[0]
        gN = [scipy.integrate.quad(integrand, vmin, np.inf)[0]
              for vmin in vmins]
        gN = np.array(gN)
    return gN / N

def g_exp(vmin, v0, vesc):
    '''
    The halo integral of the exponentially truncated Maxwellian

    I found the analytical normalization factor and use it here, which speeds
    up the code but is the reason the expressions below look so complicated.
    '''
    numerator = 6. * ((-1.+np.exp((vesc-vmin)*(vesc+vmin)/v0**2.)) * v0**2. \
                      - vesc**2. + vmin**2.)
    denominator = -6.*v0**2. * vesc - 4.*vesc**3. \
                  + 3.*np.exp(vesc**2. / v0**2.) \
                    * np.sqrt(np.pi) * v0**3. * scipy.special.erf(vesc/v0)
    g = numerator / denominator
    if isinstance(vmin, (list,np.ndarray)):
        isesc = vmin>=vesc
        g[isesc] = 0.
    else:
        if vmin>=vesc:
            return 0.
    return g

###############################################################################
# Functions for fitting the halo integral
###############################################################################

def g_integrand(v, vc, N_dict, d, e, h, j, k):
    v0 = d * (vc / 100.) ** e
    vdamp = h * (vc / 100.) ** j
    pN = pN_smooth_step_max(v, v0, vdamp, k)
    N = N_dict[vc]
    return pN / v / N

def calc_g_i(i, vmins, vcircs, N_dict, d, e, h, j, k):
    '''
    Calculate a single g given the array of vmins and vcircs and the index i
    of those at which we want to evaluate g.

    The function returns both g and the index so we can make sure that the 
    results are in order when parallel processing.

    Returns
    -------
    gi: np.ndarray, shape = (2,)
        gi[0]: the value of g
        gi[1]: the index evaluated
    '''
    assert len(vmins) == len(vcircs)
    vmin = vmins[i]
    vc = vcircs[i]
    g = scipy.integrate.quad(g_integrand, vmin, np.inf,
                             args = (vc, N_dict, d, e, h, j, k),
                             epsabs=0)[0]
    gi = np.array([g, i])
    return gi 

def normalize(pN_function, args):
    with warnings.catch_warnings():
        warnings.filterwarnings(
                'ignore', 
                category=scipy.integrate.IntegrationWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        N = scipy.integrate.quad(
                pN_function, 
                0., 
                np.inf,
                args, 
                epsabs=0
        )[0]
    return N

def calc_gs(vmins, vcircs, d, e, h, j, k, parallel=False):
    '''
    Calculate the value of the halo integral given vmin and circular
    velocity
    '''
    assert len(vmins) == len(vcircs)
    vcircs_set = np.array(list(set(vcircs)))
    v0s_set = d * (vcircs_set / 100.) ** e
    vdamps_set = h * (vcircs_set / 100.) ** j
    N_dict = {vc: normalize(pN_smooth_step_max, (v0, vdamp, k)) 
              for vc, v0, vdamp in zip(vcircs_set, v0s_set, vdamps_set)}
    if parallel:
        pool = mp.Pool(mp.cpu_count())
        gi_s = [pool.apply_async(calc_g_i, 
                                 args=(i, vmins, vcircs, N_dict,
                                       d, e, h, j, k))
                for i in range(len(vmins))]
        pool.close()
        gs, indices = np.array([gi.get() for gi in gi_s]).T
        inorder = np.all(np.diff(indices) >= 0.)
        if not inorder:
            raise ValueError('That thing you were hoping wouldn\'t happen '
                             'happened.')
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    'ignore', 
                    category=scipy.integrate.IntegrationWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            gs = [scipy.integrate.quad(g_integrand, vmin, np.inf,
                                       args = (vc, N_dict, d, e, h, j, k),
                                       epsabs = 0)[0]
                  for vmin, vc in zip(vmins, vcircs)]
        gs = np.array(gs)
    gs = np.log10(gs)
    return gs

def fit_g(galaxy='discs', limit=None, update_values=False, parallel=False):
    '''
    Fit the parameters for the sigmoid damped model (`smooth_step_max`) 
    *************************************
    ** based on the LOG halo integral. **
    *************************************
    '''
    import dm_den
    df = dm_den.load_data('dm_stats_dz1.0_20230626.h5')
    pdfs = copy.deepcopy(pdfs_v)
    pdfs.pop('m12z')
    pdfs.pop('m12w')

    if galaxy != 'discs':
        pdfs = {galaxy: pdfs[galaxy]}

    for gal in pdfs:
        dict_gal = pdfs[gal]
        bins = dict_gal['bins']
        vc_gal = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        dict_gal['vcirc'] = np.repeat(vc_gal, len(dict_gal['ps']))
        dict_gal['vesc'] = np.repeat(vesc_dict[gal]['ve_avg'],
                                     len(dict_gal['ps']))
        integrand = dict_gal['ps'] / dict_gal['vs']
        gs_gal = [scipy.integrate.simps(integrand[i:], 
                                        dict_gal['vs'][i:])
                  for i in range(len(integrand))]
        dict_gal['gs'] = np.array(gs_gal)

    ps_truth = np.array([pdfs[galname]['ps']
                         for galname in pdfs]).flatten()
    vs_truth = np.array([pdfs[galname]['vs']
                         for galname in pdfs]).flatten()
    gs_truth = np.concatenate([pdfs[gal]['gs'] for gal in pdfs]) 
    vcircs = np.array([pdfs[galname]['vcirc']
                       for galname in pdfs]).flatten()
    is_zero = gs_truth == 0.
    ps_truth = ps_truth[~is_zero]
    vs_truth = vs_truth[~is_zero]
    gs_truth = gs_truth[~is_zero]
    vcircs = vcircs[~is_zero]

    if limit is not None:
        too_small = ps_truth < limit
        ps_truth = ps_truth[~too_small]
        vs_truth = vs_truth[~too_small]
        gs_truth = gs_truth[~too_small]
        vcircs = vcircs[~too_small]

    #gs_hat = calc_gs(vs_truth, vcircs, 115., 0.928, 390., 
    #                 0.27, 0.03, parallel=parallel)

    model = lmfit.model.Model(calc_gs, independent_vars=['vmins', 'vcircs'],
                              #nan_policy = 'omit')
                              nan_policy = 'propagate',
                              parallel = parallel)
    params = model.make_params()
    params['d'].set(value=138.767313, vary=True, min=20., max=600.)
    params['e'].set(value=0.78734935, vary=True, min=0.01)
    params['h'].set(value=246.750219, vary=True, min=20., max=750.)
    params['j'].set(value=0.68338094, vary=True, min=0.01)
    params['k'].set(value=0.03089876, vary=True, min=0.001, max=1.)
    result = model.fit(np.log10(gs_truth), params, vmins=vs_truth, 
                       vcircs=vcircs)

    if update_values:
        covar = {'covar': result.covar}
        dict_gfit = {p: result.params[p].value for p in result.params.keys()}
        dict_gfit = dict_gfit | covar
        with open(paths.data + 'data_raw_gfit.pkl', 'wb') as f:
            pickle.dump(dict_gfit,
                        f, pickle.HIGHEST_PROTOCOL)

    return result 

###############################################################################
# Functions for fitting l and m to find the vesc_hat(vc) for each galaxy so 
# that their halo
# integrals are cut properly
###############################################################################

def chopped_integrand(v, vc, N_dict, d, e, h, j, k, l, m):
    v0 = d * (vc / 100.) ** e
    vdamp = h * (vc / 100.) ** j
    vesc_hat = l * (vc / 100.) ** m
    N = N_dict[vc]
    pN = pN_max_double_hard(v, v0, vdamp, k, vesc_hat)
    return pN / v / N

def calc_cut_integral_i(i, vmins, vcircs, N_dict, d, e, h, j, k, l, m):
    if len(vmins) != len(vcircs):
        raise ValueError('vmins and vcircs should be the same length.')
    vmin = vmins[i]
    vc = vcircs[i]
    g = scipy.integrate.quad(chopped_integrand, vmin, np.inf,
                             args=(vc, N_dict, d, e, h, j, k, l, m),
                             epsabs=0)[0]
    gi = np.array([g, i])
    return gi

def calc_cut_integral(vmins, vcircs, d, e, h, j, k, l, m, parallel=False):
    if len(vmins) != len(vcircs):
        raise ValueError('vmins and vcircs should be the same length.')
    def normalize(vc):
        v0 = d * (vc / 100.) ** e
        vdamp = h * (vc / 100.) ** j
        vesc_hat = l * (vc / 100.) ** m
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    'ignore', 
                    category=scipy.integrate.IntegrationWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            N = scipy.integrate.quad(pN_max_double_hard, 0., np.inf,
                                     (v0, vdamp, k, vesc_hat), epsabs=0)[0]
        return N
    vcircs_set = np.array(list(set(vcircs)))
    N_dict = {vc: normalize(vc) for vc in vcircs_set}
    if parallel:
        pool = mp.Pool(mp.cpu_count())
        gi_s = [pool.apply_async(calc_cut_integral_i,
                                 args=(i, vmins, vcircs, N_dict, 
                                       d, e, h, j, k, l, m))
                for i in range(len(vmins))]
        pool.close()
        gs, indices = np.array([gi.get() for gi in gi_s]).T
        inorder = np.all(np.diff(indices) >= 0.)
        if not inorder:
            raise ValueError('That thing you were hoping wouldn\'t happen '
                             'happened.')
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    'ignore',
                    category=scipy.integrate.IntegrationWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            gs = [scipy.integrate.quad(chopped_integrand, vmin, np.inf,
                                       args=(vc, N_dict, 
                                             d, e, h, j, k, l, m),
                                       epsabs=0)[0]
                  for vmin, vc in zip(vmins, vcircs)]
        gs = np.array(gs)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        log_gs = np.log10(gs)
    isnan = np.isinf(log_gs)
    log_gs[isnan] = -9.
    return log_gs

def fit_second_cut(df_source, limit=None,
                   parallel=False, update_values=False):
    '''
    This was an attempt to find a universal way of predicting vcut based on
    minimizing the sse of the log halo integral. However, it didn't work. It's
    not possible to do it this way, because it causes some galaxies to be cut
    so early that the sse gets driven up to the point where the lowest sse
    happens when there's no cut at all.
    '''
    import dm_den
    with open(paths.data + '/data_raw.pkl', 'rb') as f:
        speed_dist_params = pickle.load(f)
    df = dm_den.load_data(df_source)
    pdfs = copy.deepcopy(pdfs_v)
    for gal in ['m12z', 'm12w']:
        del pdfs[gal]

    for gal in pdfs:
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        pdfs[gal]['gs'] = numeric_halo_integral(pdfs[gal]['bins'],
                                                pdfs[gal]['ps'])
        pdfs[gal]['vcirc'] = np.repeat(vc, len(pdfs[gal]['gs']))

    vmins = np.concatenate([pdfs[gal]['vs'] for gal in pdfs])
    gs_truth = np.concatenate([pdfs[gal]['gs'] for gal in pdfs])
    vcircs = np.concatenate([pdfs[gal]['vcirc'] for gal in pdfs])

    if limit is not None:
        too_small = gs_truth < limit
        vmins = vmins[~too_small]
        gs_truth = gs_truth[~too_small]
        vcircs = vcircs[~too_small]
    else:
        is_zero = gs_truth == 0.
        vmins = vmins[~is_zero]
        gs_truth = gs_truth[~is_zero]
        vcircs = vcircs[~is_zero]

    model = lmfit.model.Model(calc_cut_integral, 
                              independent_vars=['vmins', 'vcircs', 'd', 'e',
                                                'h', 'j', 'k'],
                              #nan_policy='propagate',
                              nan_policy='omit',
                              parallel=parallel)
    params = model.make_params()
    params['l'].set(value=275., vary=True, min=100., max=700.)
    params['m'].set(value=1., vary=True, min=0.01)
    result = model.fit(np.log10(gs_truth), params, 
                       method='nelder',
                       vmins=vmins, vcircs=vcircs,
                       d=speed_dist_params['d'], e=speed_dist_params['e'], 
                       h=speed_dist_params['h'],
                       j=speed_dist_params['j'], k=speed_dist_params['k'])

    if update_values:
        covar = {'covar': result.covar}
        dict_gfit = {p: result.params[p].value for p in result.params.keys()}
        dict_gfit = dict_gfit | covar
        with open(paths.data + 'params_vesc_hat.pkl', 'wb') as f:
            pickle.dump(dict_gfit,
                        f, pickle.HIGHEST_PROTOCOL)

    return result

###############################################################################
# For indv gal
###############################################################################

def chopped_integrand_gal(v, vc, N_dict, d, e, h, j, k, vesc):
    v0 = d * (vc / 100.) ** e
    vdamp = h * (vc / 100.) ** j
    N = N_dict[vc]
    pN = pN_max_double_hard(v, v0, vdamp, k, vesc_hat)
    return pN / v / N

def calc_cut_integral_i_gal(i, vmins, vc, N_dict, d, e, h, j, k, vesc):
    if len(vmins) != len(vcircs):
        raise ValueError('vmins and vcircs should be the same length.')
    vmin = vmins[i]
    g = scipy.integrate.quad(chopped_integrand, vmin, np.inf,
                             args=(vc, N_dict, d, e, h, j, k, l, m),
                             epsabs=0)[0]
    gi = np.array([g, i])
    return gi

def calc_cut_integral_gal(vmins, vc, d, e, h, j, k, vesc, parallel=False):
    if len(vmins) != len(vcircs):
        raise ValueError('vmins and vcircs should be the same length.')
    def normalize(vc):
        v0 = d * (vc / 100.) ** e
        vdamp = h * (vc / 100.) ** j
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    'ignore', 
                    category=scipy.integrate.IntegrationWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            N = scipy.integrate.quad(pN_max_double_hard, 0., np.inf,
                                     (v0, vdamp, k, vesc), epsabs=0)[0]
        return N
    N_dict = {vc: normalize(vc)}
    if parallel:
        pool = mp.Pool(mp.cpu_count())
        gi_s = [pool.apply_async(calc_cut_integral_i_gal,
                                 args=(i, vmins, vc, N_dict, 
                                       d, e, h, j, k, vesc))
                for i in range(len(vmins))]
        pool.close()
        gs, indices = np.array([gi.get() for gi in gi_s]).T
        inorder = np.all(np.diff(indices) >= 0.)
        if not inorder:
            raise ValueError('That thing you were hoping wouldn\'t happen '
                             'happened.')
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    'ignore',
                    category=scipy.integrate.IntegrationWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            gs = [scipy.integrate.quad(chopped_integrand_gal, vmin, np.inf,
                                       args=(vc, N_dict, 
                                             d, e, h, j, k, vesc),
                                       epsabs=0)[0]
                  for vmin in vmins]
        gs = np.array(gs)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        log_gs = np.log10(gs)
    try:
        isnan = np.isinf(log_gs)
    except RuntimeWarning:
        print(log_gs)
    log_gs[isnan] = -9.
    return log_gs

def fit_vesc_indv(df_source, limit=1.e-5,
                  update_values=False):
    import dm_den
    with open(paths.data + '/data_raw.pkl', 'rb') as f:
        speed_dist_params = pickle.load(f)
    df = dm_den.load_data(df_source)
    pdfs = copy.deepcopy(pdfs_v)
    for gal in ['m12z', 'm12w']:
        pdfs.pop(gal)

    vesc_dict = {}

    for gal in pdfs:
        print('Fitting {0:s}'.format(gal))
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        gs_truth = numeric_halo_integral(pdfs[gal]['bins'],
                                         pdfs[gal]['ps'])
        vmins = pdfs[gal]['vs']

        if limit is not None:
            too_small = gs_truth < limit
            vmins = vmins[~too_small]
            gs_truth = gs_truth[~too_small]
        else:
            is_zero = gs_truth == 0.
            vmins = vmins[~is_zero]
            gs_truth = gs_truth[~is_zero]

        v0 = speed_dist_params['d'] * (vc / 100.) ** speed_dist_params['e']
        vdamp = speed_dist_params['h'] * (vc / 100.) ** speed_dist_params['j']

        def calc_log_halo_integral(vmins, vesc):
            gs = calc_g_general(vmins, pN_max_double_hard,
                                (v0, vdamp, speed_dist_params['k'], vesc))
            is_zero = gs == 0.
            gs[is_zero] = 1.e-8
            log_gs = np.log10(gs)
            isinf = np.isinf(log_gs)
            return log_gs

        model = lmfit.model.Model(calc_log_halo_integral, 
                                  #nan_policy='propagate',
                                  nan_policy='omit')
        params = model.make_params()
        params['vesc'].set(value=500., vary=True, min=300., max=700.)

        result = model.fit(np.log10(gs_truth), params, 
                           method='nelder',
                           vmins=vmins)

        vesc_dict[gal] = result.params['vesc'].value

    if update_values:
        with open(paths.data + 'vesc_ideal.pkl', 'wb') as f:
            pickle.dump(vesc_dict,
                        f, pickle.HIGHEST_PROTOCOL)

    return vesc_dict

###############################################################################
# Fit and plot speed distributions
###############################################################################

def fit_v0(gals='discs', show_exp=False, tgt_fname=None):
    '''
    Plot a simple Maxwellian with the best-fit v0
    '''
    import dm_den
    import dm_den_viz
    df = dm_den.load_data('dm_stats_dz1.0_20230626.h5')
    if gals == 'discs':
        df = df.drop(['m12w', 'm12z'])
    elif isinstance(gals, (list, np.ndarray)):
        df = df.loc[gals]
    else:
        raise ValueError('Unexpected value provided for gals arg')
    
    #if gals == 'discs':
    #    figsize = (19., 12.)
    #    Nrows = 3
    #    Ncols = 4
    #else:
    #    Ncols = min(len(gals), 4)
    #    Nrows = math.ceil(len(gals) / Ncols)
    #xfigsize = 4.5 * Ncols + 1.
    #yfigsize = 3.7 * Nrows + 1. 
    #fig,axs=plt.subplots(Nrows, Ncols, figsize=(xfigsize, yfigsize), 
    #                     sharey='row',
    #                     sharex=True, dpi=140)
    #axs=axs.ravel()
    #fig.subplots_adjust(wspace=0.,hspace=0.)
    
    fig, axs = dm_den_viz.setup_multigal_fig(gals)

    pbar = ProgressBar()
    
    def sse_max_v0_vesc(params, vs_truth, ps_truth):
        v0 = params['v0'].value
        #print(v0)
        vesc = params['vesc'].value
        k = params['k'].value
        with np.errstate(divide='ignore'):
            ps_predicted = smooth_step_max(
                                        vs_truth,
                                        v0, 
                                        vesc,
                                        k)
        resids = ps_predicted - ps_truth
        return resids
    
    def resids_exp_max(params, vs_truth, ps_truth):
        v0 = params['v0'].value
        vesc = params['vesc'].value
        with np.errstate(divide='ignore'):
            ps_predicted = exp_max(vs_truth,
                                   v0,
                                   vesc)
        resids = ps_predicted - ps_truth
        return resids
        
    vesc_fits = {}
    
    vcut_dict = dm_den.load_vcuts('lim_fit', df)

    for i, gal in enumerate(pbar(df.index)):
        pdf = pdfs_v[gal]
        bins = pdf['bins']
        vs_truth = (bins[1:] + bins[:-1]) / 2.
        vs_postfit = np.linspace(0., 750., 500)
        ps_truth = pdf['ps']
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        vesc_Phi = vesc_dict[gal]['ve_avg']
        i0 = np.argmax(ps_truth)
        v0_data = vs_truth[i0]
        axs[i].axvline(v0_data, c='grey', lw=1., alpha=0.5)
        axs[i].axhline(ps_truth[i0], c='grey', lw=1., alpha=0.5)
    
        #######################################################################
        p = lmfit.Parameters()
        p.add('v0', value=300., vary=True, min=100., max=400.)
        p.add('vesc', value=vcut_dict[gal], vary=False)
        p.add('k', value=np.inf, vary=False)
        #p.add('vesc', value=np.inf, vary=False)
        #p.add('k', value=0.0309, vary=False, min=0.0001, max=1.)
    
        res_v0_vesc = lmfit.minimize(sse_max_v0_vesc, p, 
                                     method='nelder', 
                                     args=(vs_truth, ps_truth),
                                     nan_policy='omit', 
                                     #niter=300
                                     )
        vesc_fits[gal] = res_v0_vesc.params['vesc'].value
        
        axs[i].stairs(ps_truth, bins, color='grey')
        axs[i].plot(
            vs_postfit, 
            smooth_step_max(
                vs_postfit, 
                res_v0_vesc.params['v0'], 
                res_v0_vesc.params['vesc'],
                res_v0_vesc.params['k']),
            label='best fit $v_0$\ncut @ $\hat{v}_\mathrm{lim}(v_\mathrm{c})$',
            zorder=10)
        #######################################################################
        
        #######################################################################
        # Plot simple Maxwellian with best-fit v0, without a cut
        #######################################################################
        p_v0_uncut = lmfit.Parameters()
        p_v0_uncut.add('v0', value=300., vary=True, min=100., max=400.)
        p_v0_uncut.add('vesc', value=np.inf, vary=False)
        p_v0_uncut.add('k', value=np.inf, vary=False)
        res_v0_uncut = lmfit.minimize(sse_max_v0_vesc, p_v0_uncut, 
                                      method='nelder', 
                                      args=(vs_truth, ps_truth),
                                      nan_policy='omit', 
                                      #niter=300
                                     )
        axs[i].plot(vs_postfit,
                    smooth_step_max(vs_postfit,
                                    res_v0_uncut.params['v0'],
                                    res_v0_uncut.params['vesc'],
                                    res_v0_uncut.params['k']),
                    label='best fit $v_0$')
        #######################################################################

        # Plot the naive Maxwellian
        axs[i].plot(vs_postfit,
                    smooth_step_max(vs_postfit, vc, np.inf, np.inf),
                    label='$v_0=v_\mathrm{c}$')

        # Force v0 to be at the data's peak
        axs[i].plot(vs_postfit,
                    smooth_step_max(vs_postfit, v0_data, vcut_dict[gal], 
                                    np.inf),
                    label=('correct $v_0$'
                           '\ncut @ $\hat{v}_\mathrm{lim}(v_\mathrm{c})$'))

        if show_exp:
            del(p['k'])
            #p['vesc'].set(value = vesc_dict[gal]['ve_avg'], vary=False)
            res_exp = lmfit.minimize(resids_exp_max, p, method='nelder',
                                     args=(vs_truth, ps_truth), 
                                     nan_policy='omit')
            axs[i].plot(vs_postfit,
                        exp_max(vs_postfit, 
                                res_exp.params['v0'],
                                res_exp.params['vesc']))

        axs[i].grid(False)
        
        loc=[0.97,0.96]
        kwargs_txt = dict(fontsize=13., xycoords='axes fraction',
                      va='top', ha='right', 
                      bbox=dict(facecolor='white', alpha=0.8, 
                                edgecolor='none', pad=0.))
        axs[i].annotate(gal, loc,
                        **kwargs_txt)
        loc[1] -= 0.13
        kwargs_txt['fontsize'] = 8.
        axs[i].annotate(#'$v_\mathrm{{esc}}'
                        #'={0:0.0f}\,\mathrm{{km\,s^{{-1}}}}$\n'
                        '$\hat{{v}}_\mathrm{{lim}}'
                        '={1:0.0f}\,\mathrm{{km\,s^{{-1}}}}$\n'
                        '$v_\mathrm{{0,best}}={3:0.0f}$\n'
                        #'$k={6:0.4f}$\n'
                        '$\chi^2={2:0.2e}$\n'
                        #'N$_\mathrm{{eval}}={5:0.0f}$'
                        .format(vesc_Phi, 
                                res_v0_vesc.params['vesc'].value,
                                res_v0_vesc.chisqr, 
                                res_v0_vesc.params['v0'].value,
                                None,
                                res_v0_vesc.nfev,
                                res_v0_vesc.params['k'].value,
                               ),
                        loc, **kwargs_txt)
        if show_exp:
            loc[1] -= 0.2
            axs[i].annotate('$\chi^2_\mathrm{{exp}}={0:0.2e}$\n'
                            .format(res_exp.chisqr),
            loc, **kwargs_txt)

    dm_den_viz.label_axes(axs, fig, gals)

    axs[-1].legend(loc='upper center', 
                   bbox_to_anchor=(.5, -0.02),
                   bbox_transform=fig.transFigure, ncols=4)

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=140)

    plt.show()

    return vesc_fits

def fit_vdamp(df_source, gals='discs', 
              vcut_type=None,
              show_max=False,
              show_max_fit=False,
              show_exp=False, 
              show_mao_fixed=False, 
              show_mao_free=False, 
              show_rms=False,
              show_resids=True,
              show_vescs=True,
              tgt_fname=None, sigmoid_damped_eqnum=None,
              xtickspace=None):
    '''
    Plot the best posible distributions, individually fitting vdamp and v0 for
    each galaxy

    Parameters
    ----------
    df_source: str
        File name for the analysis results to use
    gals: str or list-like of str
        Galaxies to plot
    show_max: bool
        If True, include the naive Maxwellian with v0=vc
    show_max_fit: bool
        If True, include the Maxwellian with the best fit v0, cut at vcut_type
    show_exp: bool
        If True, include the exponentially cutoff form from Macabe 2010 and 
        Lacroix et al. 2020.
    show_mao_fixed: bool
        If True, include the best-fit parameterization from Mao et al. 2013
        with vesc = vcut(Phi)
    show_mao_free: bool
        If True, include the best-fit parameterization from Mao et al. 2013
        where we also let vesc be a free parameter
    vcut_type: {'lim_fit', 'lim', 'vesc_fit', 'veschatphi', 'ideal'} 
               default None
        Specifies how to determine the speed distribution cutoff.
            lim: The true escape speed v_esc -- The speed of the fastest DM 
                particle in 
                the solar ring.
            lim_fit: \hat{v}_{esc}(v_c) -- A regression of v_esc to vc
            veschatphi: \hat{v}_{esc}(Phi) -- An estimate of the escape speed
                based on gravitational potential Phi
            vesc_fit: \hat{v}_{esc}(Phi-->v_c) -- A regression of veschatphi
                to vc
            ideal: The final cutoff speed that would optimize the galaxy's halo
                integral's fit when calculating the halo integral of a 
                fitting.max_double_hard model
    show_rms: bool
        If true, show information about root mean square deviations of the
        distribution models from the data
    tgt_fname: str
        File name with which to save the plot. Default, None, is to not save
        the plot.

    Return
    ------
    vdamp_fits: dict
        Dictionary keyed by galaxy of the best fit vdamps
    '''
    if (show_mao_free or show_mao_fixed or show_exp) and vcut_type is None:
        raise ValueError('You must specify a vcut_type if you want to show'
                         ' a distribution with a cut.')
    import dm_den
    import dm_den_viz
    df = dm_den.load_data(df_source)
    if gals == 'discs':
        df = df.drop(['m12w', 'm12z'])
    elif isinstance(gals, (list, np.ndarray)):
        df = df.loc[gals]
    else:
        raise ValueError('Unexpected value provided for gals arg')
    Ngals = len(df)    

    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        results_universal = pickle.load(f)
    vcut_dict = dm_den.load_vcuts(vcut_type, df)

    def sse_max_v0_vesc(params, vs_truth, ps_truth):
        v0 = params['v0'].value
        #print(v0)
        vdamp = params['vdamp'].value
        k = params['k'].value
        with np.errstate(divide='ignore'):
            ps_predicted = smooth_step_max(
                                        vs_truth,
                                        v0, 
                                        vdamp,
                                        k)
        resids = ps_predicted - ps_truth
        return resids
    
    def resids_exp_max(params, vs_truth, ps_truth):
        v0 = params['v0'].value
        vcut = params['vcut'].value
        with np.errstate(divide='ignore'):
            ps_predicted = exp_max(vs_truth,
                                   v0,
                                   vcut)
        resids = ps_predicted - ps_truth
        return resids
        
    vdamp_fits = {}
    
    sse_mao_fixed = 0.
    sse_mao_fixed_tail = 0.
    sse_mao_free = 0.
    sse_mao_free_tail = 0.
    sse = 0. # SSE for the smooth_step_max (our method)
    sse_tail = 0.
    d = 2 # Number of digits to show in the RMS
    O_rms = 1.e-3 # RMS order of magnitude to show
    N = 0 # Number of datapoints evaluated, for use in aggregate RMS
    N_tail = 0

    fig, axs = dm_den_viz.setup_multigal_fig(gals, show_resids=show_resids)

    # vesc's estimated from gravitational potential
    vescphi_dict = dm_den.load_vcuts('veschatphi', df)

    pbar = ProgressBar()
    for i, gal in enumerate(pbar(df.index)):
        pdf = pdfs_v[gal]
        bins = pdf['bins']
        vs_truth = (bins[1:] + bins[:-1]) / 2.
        N += len(vs_truth) # for use in calculating aggregate rms
        vs_postfit = np.linspace(0., 750., 500)
        ps_truth = pdf['ps']
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        vcut = vcut_dict[gal]
        sigma_truth = df.loc[gal, 'disp_dm_disc_cyl']
        i0 = np.argmax(ps_truth)
        v0_truth = vs_truth[i0]
    
        # Draw a line at the 84th %ile:
        # P = probability of being within 1. std devs:
        P = scipy.special.erf(1. / np.sqrt(2.)) 
        percentile = P + (1 - P) / 2.
        cdf = scipy.integrate.cumulative_trapezoid(np.insert(ps_truth, 0, 0.), 
                                                   np.insert(vs_truth, 0, 0.))
        in_tail = cdf >= percentile

        p = lmfit.Parameters()
        p.add('v0', value=300., vary=True, min=100., max=400.)
        p.add('vdamp', value=470., vary=True, min=250., max=600.)
        p.add('k', value=results_universal['k'], vary=False, min=0.0001, max=1.)
    
        res_v0_vesc = lmfit.minimize(sse_max_v0_vesc, p, 
                                     method='nelder', 
                                     args=(vs_truth, ps_truth),
                                     nan_policy='omit', 
                                     #niter=300
                                    )
        vdamp_fits[gal] = res_v0_vesc.params['vdamp'].value
        _ = [res_v0_vesc.params[key] 
             for key in ['v0', 'vdamp', 'k']]

        rms_err, sse_add = calc_rms_err(vs_truth, ps_truth, smooth_step_max,
                                        args=[res_v0_vesc.params[key] 
                                              for key in ['v0', 'vdamp', 'k']])
        sse += sse_add
        N_tail += np.sum(in_tail)
        _, sse_tail_add = calc_rms_err(vs_truth[in_tail], ps_truth[in_tail],
                                       smooth_step_max,
                                       args=[res_v0_vesc.params[key] 
                                             for key in ['v0', 'vdamp', 'k']])
        sse_tail += sse_tail_add
        rms_txt_sigmoid = staudt_utils.mprint(rms_err/O_rms, d=d, 
                                      show=False).replace('$','')
        
        # Plot the naive Maxwellian
        if show_max:
            axs[i].plot(
                vs_postfit,
                smooth_step_max(
                    vs_postfit,
                    vc,
                    np.inf,
                    np.inf
                ),
                label='Maxwellian, $v_0=v_{\\rm c}$', 
                c=dm_den_viz.max_naive_color
            )

        # Plot the best-fit simple Maxwellian
        if show_max_fit:
            model_max = lmfit.model.Model(smooth_step_max,
                                          independent_vars=['v', 'vdamp', 'k'])
            params_max = model_max.make_params()
            params_max['v0'].set(value=220., vary=True, min=100., max=400.)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                        'ignore', 
                        category=scipy.integrate.IntegrationWarning)
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                result_max = model_max.fit(ps_truth, params_max,
                                           v=vs_truth, vdamp=vcut, k=np.inf)
            axs[i].plot(vs_postfit,
                        smooth_step_max(vs_postfit, result_max.params['v0'],
                                        vcut, np.inf),
                        label='fit, Maxwellian, cut @ {0:s}'.format(
                            dm_den_viz.vcut_labels[vcut_type]), 
                        c=dm_den_viz.max_fit_color)

        # Plot the data
        axs[i].stairs(ps_truth, bins, color='k', label='data')

        # Set label for the line from this work
        if sigmoid_damped_eqnum is not None:
            label_this = 'fit, Eq. ' + str(sigmoid_damped_eqnum)
        elif show_mao_fixed or show_mao_free or show_exp:
            label_this = 'fit, sigmoid damped'
        else:
            label_this = 'fit'

        axs[i].plot(
            vs_postfit, 
            smooth_step_max(
                vs_postfit, 
                res_v0_vesc.params['v0'], 
                res_v0_vesc.params['vdamp'],
                res_v0_vesc.params['k']),
                label=label_this, color='C2')
        if show_exp:
            params_exp = lmfit.Parameters()
            params_exp.add('v0', value=220., vary=True, min=100., max=400.)
            params_exp.add('vcut', value=470., vary=True, min=250., max=1000.)
            res_exp = lmfit.minimize(resids_exp_max, params_exp, 
                                     method='nelder',
                                     args=(vs_truth, ps_truth), 
                                     nan_policy='omit')
            axs[i].plot(vs_postfit,
                        exp_max(vs_postfit, 
                                res_exp.params['v0'],
                                res_exp.params['vcut']),
                        label='fit, exp trunc')

            params_exp['vcut'].set(value=vcut, 
                                   vary=False, 
                                   max=np.inf)
            res_exp_fixed_vcut = lmfit.minimize(
                                     resids_exp_max, params_exp, 
                                     method='nelder',
                                     args=(vs_truth, ps_truth), 
                                     nan_policy='omit')
            axs[i].plot(vs_postfit,
                        exp_max(vs_postfit, 
                                res_exp_fixed_vcut.params['v0'],
                                res_exp_fixed_vcut.params['vcut']),
                        label='fit, exp trunc @ $v_\mathrm{esc}(\Phi)$',
                        color='C4')

        if show_mao_fixed or show_mao_free:
            # Testing Mao parameterization
            model_mao = lmfit.Model(mao)
            params_mao = model_mao.make_params()
            params_mao['v0'].set(value=vc, vary=True, min=100., max=400.)
            params_mao['vesc'].set(value=vcut, vary=False, min=vc, max=900.)
            params_mao['p'].set(value=1., vary=True, min=0.)

            if show_mao_fixed:
                result_mao_fixed = model_mao.fit(ps_truth, params_mao,
                                                 v=vs_truth,
                                                 method='nelder')
                if show_mao_free:
                    fixed_label = 'Mao, fixed $v_\mathrm{esc}(\Phi)$'
                else:
                    fixed_label = 'fit, Mao+13'
                axs[i].plot(vs_postfit,
                            mao(vs_postfit,
                                result_mao_fixed.params['v0'].value,
                                result_mao_fixed.params['vesc'].value,
                                result_mao_fixed.params['p'].value),
                            label=fixed_label,
                            color='C1', zorder=1)
                rms_err_mao_fixed, sse_mao_fixed_add = calc_rms_err(
                        vs_truth, ps_truth, mao,
                        args=[result_mao_fixed.params[key] 
                              for key in result_mao_fixed.params])
                sse_mao_fixed += sse_mao_fixed_add
                _, sse_mao_fixed_tail_add = calc_rms_err(
                        vs_truth[in_tail], 
                        ps_truth[in_tail],
                        mao,
                        args=[result_mao_fixed.params[key] 
                              for key in result_mao_fixed.params])
                sse_mao_fixed_tail += sse_mao_fixed_tail_add
                rms_txt_mao_fixed = staudt_utils.mprint(
                        rms_err_mao_fixed/O_rms, d=d, 
                        show=False).replace('$','')

            if show_mao_free:
                params_mao['vesc'].set(vary=True)
                result_mao_free = model_mao.fit(ps_truth, params_mao,
                                           v=vs_truth,
                                           method='nelder')
                print(result_mao_free.params['p'].value)
                axs[i].plot(vs_postfit,
                            mao(vs_postfit,
                                result_mao_free.params['v0'].value,
                                result_mao_free.params['vesc'].value,
                                result_mao_free.params['p'].value),
                            label='Mao, free $v_\mathrm{esc}$',
                            color='C4')
                rms_err_mao_free, sse_mao_free_add = calc_rms_err(
                        vs_truth, ps_truth, mao,
                        args=[result_mao_free.params[key] 
                              for key in result_mao_free.params])
                sse_mao_free += sse_mao_free_add
                _, sse_mao_free_tail_add = calc_rms_err(
                        vs_truth[in_tail], ps_truth[in_tail],
                        mao,
                        args=[result_mao_free.params[key] 
                              for key in result_mao_free.params])
                sse_mao_free_tail += sse_mao_free_tail_add
                rms_txt_mao_free = staudt_utils.mprint(
                        rms_err_mao_free/O_rms, d=d, 
                        show=False).replace('$','')

        if show_vescs:
            # Draw vesc(Phi) line
            vesc = vescphi_dict[gal]
            axs[i].axvline(vesc, ls='--', alpha=0.8, color='k')
            trans = mpl.transforms.blended_transform_factory(axs[i].transData,
                                                             axs[i].transAxes)
            if vesc >= vcut:
                vesc_adj = 20.
                vesc_ha = 'left'
                vcut_adj = 0.
                vcut_ha = 'right'
            else:
                vesc_adj = 0.
                vesc_ha = 'right'
                vcut_adj = 20. 
                vcut_ha = 'left'
            if vcut_type == 'vesc_fit':
                veschatphi_label_y = 0.8
                veschatphi_va = 'top'
                vcut_label_y = 0.8
                vcut_va = 'top'
            else:
                veschatphi_label_y = 0.4
                veschatphi_va = 'baseline'
                vcut_label_y = 0.4
                vcut_va = 'baseline'
            axs[i].text(vesc + vesc_adj, veschatphi_label_y, 
                        dm_den_viz.vcut_labels['veschatphi'], 
                        transform=trans,
                        fontsize=15., rotation=90., color='k', 
                        horizontalalignment=vesc_ha,
                        verticalalignment=veschatphi_va)

            # Draw vcut line
            #vlim = dm_den.load_vcuts('lim', df)[gal]
            #axs[i].axvline(vlim, ls='--', alpha=0.5, color='C0')
            axs[i].axvline(vcut, ls='--', alpha=0.5, color='grey')
            trans = mpl.transforms.blended_transform_factory(axs[i].transData,
                                                             axs[i].transAxes)
            axs[i].text(vcut + vcut_adj, vcut_label_y, 
                        dm_den_viz.vcut_labels[vcut_type], 
                        transform=trans,
                        fontsize=15., rotation=90., color='gray', 
                        horizontalalignment=vcut_ha, verticalalignment=vcut_va)

        axs[i].grid(False)

        # Make ticks on both sides of the x-axis:
        axs[i].tick_params(axis='x', direction='inout', length=6)

        order_of_mag = -3
        dm_den_viz.make_sci_y(axs, i, order_of_mag)

        loc=[0.97,0.96]
        if fig.Nrows == 3:
            namefs = 13.
        else:
            namefs = 16.
        kwargs_txt = dict(fontsize=namefs, xycoords='axes fraction',
                          va='top', ha='right', 
                          bbox=dict(facecolor='white', alpha=0.4, 
                                    edgecolor='none'))
        axs[i].annotate(gal, loc,
                        **kwargs_txt)
        loc[1] -= 0.16
        kwargs_txt['fontsize'] = 9.
        if show_rms:
            if sigmoid_damped_eqnum is not None:
                staudt_rms_txt = '$\mathrm{{RMS_{{{9:0.0f}}}}}={4:0.2f}$\n'
            else:
                staudt_rms_txt = '$\mathrm{{RMS_{{Staudt}}}}={4:0.2f}$\n'
            annotation = (
                          #'$v_\mathrm{{damp}}'
                          #'={1:0.0f}\,\mathrm{{km\,s^{{-1}}}}$\n'
                          #'$v_0={3:0.0f}$\n'
                          #'$k={6:0.4f}$\n'
                          staudt_rms_txt
                          #'N$_\mathrm{{eval}}={5:0.0f}$'
                         )
            if show_mao_fixed:
                if show_mao_free:
                    annotation += ('$\mathrm{{RMS_{{Mao\,fixed}}}}={7:s}$'
                                   '\n$\mathrm{{RMS_{{Mao\,free}}}}={8:s}$')
                else:
                    annotation += '$\mathrm{{RMS_{{Mao}}}}={7:0.2f}$'
            axs[i].annotate(annotation.format(
                                None, 
                                res_v0_vesc.params['vdamp'].value,
                                res_v0_vesc.chisqr, 
                                res_v0_vesc.params['v0'].value,
                                rms_err / O_rms,
                                res_v0_vesc.nfev,
                                res_v0_vesc.params['k'].value,
                                rms_err_mao_fixed / O_rms if show_mao_fixed \
                                    else None,
                                rms_txt_mao_free if show_mao_free else None,
                                sigmoid_damped_eqnum
                               ),
                            loc, **kwargs_txt)
        '''
        if show_exp:
            loc[1] -= 0.2
            axs[i].annotate('$\chi^2_\mathrm{{exp}}={0:0.2e}$\n'
                            .format(res_exp.chisqr),
            loc, **kwargs_txt)
        '''
        if Ngals <= 3 and show_resids:
            # Remove the 0 tick label because of overlap
            y0, y1 = axs[i].get_ylim()
            visible_ticks = np.array([t for t in axs[i].get_yticks() \
                                      if t>=y0 and t<=y1])
            new_ticks = visible_ticks[visible_ticks > 0.]
            axs[i].set_yticks(new_ticks)

            # Draw residual plot
            vs_resids = copy.deepcopy(vs_truth)
            vs_extend = np.linspace(vs_resids.max(), vs_postfit.max(), 20)
            vs_resids = np.append(vs_resids, vs_extend, axis=0)         
            def calc_resids(func, args):
                ps_hat = func(
                        vs_truth, 
                        *args)
                resids = ps_hat - ps_truth
                resids_extend = func(
                        vs_extend, 
                        *args)
                resids = np.append(resids, resids_extend, axis=0)
                return resids
            resids = calc_resids(
                    smooth_step_max, 
                    args=(res_v0_vesc.params['v0'].value,
                          res_v0_vesc.params['vdamp'].value,
                          res_v0_vesc.params['k'].value))
            axs[i+Ngals].plot(vs_resids, resids / 10.**order_of_mag, 
                              color='C2')
            if show_mao_fixed:
                resids_mao_fixed = calc_resids(
                        mao,
                        args=(result_mao_fixed.params['v0'].value,
                              result_mao_fixed.params['vesc'].value,
                              result_mao_fixed.params['p'].value))
                axs[i+Ngals].plot(vs_resids, 
                                  resids_mao_fixed / 10.**order_of_mag, 
                                  color='C1', zorder=1)


            axs[i+Ngals].axhline(0., linestyle='--', color='k', alpha=0.5, 
                                 lw=1.)
            axs[i+Ngals].set_ylim(-dm_den_viz.resids_lim, dm_den_viz.resids_lim)
            if i == 0:
                axs[i+Ngals].set_ylabel('resids')

        if xtickspace is not None:
            axs[i].xaxis.set_major_locator(
                    mpl.ticker.MultipleLocator(base=xtickspace))
            axs[i].xaxis.set_minor_locator(plt.NullLocator())
    dm_den_viz.label_axes(axs, fig, gals)
    trans = fig.transFigure
    if Ngals >2:
        ncol = 3
        legend_y = 0.03
    else:
        ncol = 2 
        legend_y = 0.
    axs[0].legend(loc='upper center',
                  #bbox_to_anchor=(1., -0.5),
                  bbox_to_anchor=(0.5, legend_y),
                  bbox_transform=trans, 
                  ncol=ncol, borderaxespad=1.5)
    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=350)

    plt.show()

    if show_rms:
        if show_mao_fixed:
            txt = staudt_utils.mprint(
                    np.sqrt(sse_mao_fixed / N),
                    d=d, 
                    show=False).replace('$','')
            if show_mao_free:
                display(Latex('$\mathrm{{RMS_{{Mao\,fixed}}}}={0:s}$'
                              .format(txt)))
            else:
                display(Latex('$\mathrm{{RMS_{{Mao}}}}={0:s}$'
                              .format(txt)))
        if show_mao_free:
            txt = staudt_utils.mprint(
                    np.sqrt(sse_mao_free / N),
                    d=d, 
                    show=False).replace('$','')
            display(Latex('$\mathrm{{RMS_{{Mao\,free}}}}={0:s}$' .format(txt)))
        txt = staudt_utils.mprint(
                np.sqrt(sse / N),
                d=d, 
                show=False).replace('$','')
        display(Latex('$\mathrm{{RMS_{{sigmoid\,damped}}}}={0:s}$' \
                          .format(txt)))

        print('\nBeyond the 84th %ile:')
        if show_mao_fixed:
            txt = staudt_utils.mprint(
                    np.sqrt(sse_mao_fixed_tail / N),
                    d=d, 
                    show=False).replace('$','')
            if show_mao_free:
                display(Latex('$\mathrm{{RMS_{{Mao\,fixed}}}}={0:s}$'
                              .format(txt)))
            else:
                display(Latex('$\mathrm{{RMS_{{Mao}}}}={0:s}$'
                              .format(txt)))
        if show_mao_free:
            txt = staudt_utils.mprint(
                    np.sqrt(sse_mao_free_tail / N),
                    d=d, 
                    show=False).replace('$','')
            display(Latex('$\mathrm{{RMS_{{Mao\,free}}}}={0:s}$'
                          .format(txt)))
        txt = staudt_utils.mprint(
                np.sqrt(sse_tail / N),
                d=d, 
                show=False).replace('$','')
        display(Latex('$\mathrm{{RMS_{{sigmoid\,damped}}}}={0:s}$' \
                      .format(txt)))
    return vdamp_fits

def fit_universal_no_uncert(gals='discs', method='leastsq', update_vals=False,
                            tgt_fname=None, plot=True, 
                            vary_dict = {'d': True,
                                         'e': True,
                                         'h': True,
                                         'j': True,
                                         'k': False},
                            **kwargs):
    '''
    This was how I performed the universal fit before implementing lmfit.Model.
    '''

    if tgt_fname is not None and not plot:
        raise ValueError('tgt_fname for plot image can only be specified if'
                         ' plot is True')
    import dm_den
    df = dm_den.load_data('dm_stats_dz1.0_20230626.h5').drop(['m12w', 'm12z'])
    if gals != 'discs' and not isinstance(gals, (list, np.ndarray)):
        raise ValueError('Unexpected value provided for gals arg')
    pdfs = copy.deepcopy(pdfs_v)
    pdfs.pop('m12z')
    pdfs.pop('m12w')

    for gal in pdfs:
        dict_gal = pdfs[gal]
        bins = dict_gal['bins']
        dict_gal['vcirc'] = np.repeat(
            df.loc[gal, 'v_dot_phihat_disc(T<=1e4)'],
            len(dict_gal['ps']))
        dict_gal['vesc'] = np.repeat(vesc_dict[gal]['ve_avg'],
                                     len(dict_gal['ps']))

    ps_truth = np.array([pdfs[galname]['ps']
                   for galname in pdfs]).flatten()
    vs_truth = np.array([pdfs[galname]['vs']
                   for galname in pdfs]).flatten()
    vs_postfit = np.linspace(0., 750., 300)

    vcircs = np.array([pdfs[galname]['vcirc']
                       for galname in pdfs]).flatten()
    vescs = np.array([pdfs[galname]['vesc']
                      for galname in pdfs]).flatten()

    p = lmfit.Parameters()
    p.add('d', value=1.60030613, vary=vary_dict['d'], min=0.1, max=4.)
    p.add('e', value=0.92819047, vary=vary_dict['e'], min=0.1, max=4.)
    p.add('h', value=111.783463, vary=vary_dict['h'], min=5., max=300.)
    p.add('j', value=0.27035357, vary=vary_dict['j'], min=0.05, max=4.)
    p.add('k', value=0.03089489, vary=vary_dict['k'], min=0.0001, max=1.)

    ###########################################################################
    ## Fitting 
    ###########################################################################
    def resids_dehjk(params, vcircs, vs_truth, ps_truth):
        d = params['d'].value
        e = params['e'].value
        h = params['h'].value
        j = params['j'].value
        k = params['k'].value
        with np.errstate(divide='ignore'):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=scipy.integrate.IntegrationWarning)
                ps_predicted = [smooth_step_max(
                                            v,
                                            v0 = d * vc ** e,
                                            vesc = h * vc ** j,
                                            k = k)
                                for v, vc in zip(vs_truth, vcircs)]
        ps_predicted = np.array(ps_predicted)
        resids = ps_predicted - ps_truth
        return resids

    result_dehjk = lmfit.minimize(resids_dehjk, p, method=method,
                                  args=(vcircs, vs_truth, ps_truth),
                                  **kwargs)
    if update_vals:
        data2save = {key: result_dehjk.params[key].value
                     for key in result_dehjk.params.keys()}
        stderrs = {key+'_stderr': result_dehjk.params[key].stderr
                   for key in result_dehjk.params.keys()}
        covar = {'covar': result_dehjk.covar}
        data2save = data2save | stderrs | covar #combine dictionaries
        dm_den.save_var_raw(data2save)
    ###########################################################################

    if plot:
        if gals == 'discs':
            figsize = (19., 12.)
            Nrows = 3
            Ncols = 4
        else:
            Ncols = min(len(gals), 4)
            Nrows = math.ceil(len(gals) / Ncols)
        xfigsize = 4.5 * Ncols + 1.
        yfigsize = 3.7 * Nrows + 1. 

        fig,axs=plt.subplots(Nrows, Ncols, figsize=(xfigsize, yfigsize), 
                             sharey='row',
                             sharex=True, dpi=140)
        axs=axs.ravel()
        fig.subplots_adjust(wspace=0.,hspace=0.)

        if isinstance(gals, (list, np.ndarray)):
            df = df.loc[gals]

        for i, gal in enumerate(df.index):
            vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
            axs[i].stairs(pdfs[gal]['ps'], pdfs[gal]['bins'], color='grey')
            axs[i].plot(
                vs_postfit,
                smooth_step_max(
                    vs_postfit,
                    v0 = result_dehjk.params['d'] * vc ** result_dehjk.params['e'],
                    vesc = result_dehjk.params['h'] \
                           * vc ** result_dehjk.params['j'],
                    k = result_dehjk.params['k']),
                label='fitting $v_\mathrm{damp}$', 
                #c='#00c600'
                ) 

            loc = [0.97,0.96]
            kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                              va='top', ha='right',
                              bbox=dict(facecolor='white', alpha=0.8, 
                                        edgecolor='none'))
            axs[i].annotate(gal, loc,
                            **kwargs_txt)

            axs[i].grid()

        dm_den_viz.label_axes(axs, fig, gals)

        if tgt_fname is not None:
            plt.savefig(paths.figures+tgt_fname,
                        bbox_inches='tight',
                        dpi=140)

        plt.show()

    return result_dehjk

def fit_mao(vcut_type, df_source, update_values=False):
    '''
    Find a universal d, e, and p for the Mao parameterization where
    v0 = d * vc ^ e.
    
    Parameters
    ----------
    vcut_type: {'lim_fit', 'lim', 'vesc_fit', 'vesc', 'ideal'},
               default 'lim_fit'
        Specifies how to determine the speed distribution cutoff.
    update_values: bool, default True
        Whether to save parameters to the LaTeX data and a raw .pkl file.
    '''
    import dm_den
    import dm_den_viz
    df = dm_den.load_data(df_source).drop(['m12w', 'm12z'])
    pdfs = copy.deepcopy(pdfs_v)
    galnames = list(pdfs.keys())
    for gal in ['m12z', 'm12w']:
        galnames.remove(gal)
    vcut_dict = dm_den.load_vcuts(vcut_type, df)
    for gal in galnames:
        dict_gal = pdfs[gal]
        bins = dict_gal['bins']
        dict_gal['vcirc'] = np.repeat(
            df.loc[gal, 'v_dot_phihat_disc(T<=1e4)'],
            len(dict_gal['ps']))
        dict_gal['vesc'] = np.repeat(vcut_dict[gal],
                                     len(dict_gal['ps']))

    ps_truth = np.array([pdfs[galname]['ps']
                   for galname in galnames]).flatten()
    vs_truth = np.array([pdfs[galname]['vs']
                   for galname in galnames]).flatten()
    
    N_postfit = 300
    vs_postfit = np.linspace(0., 700., N_postfit)

    vcircs = np.array([pdfs[galname]['vcirc']
                         for galname in galnames]).flatten()
    vescs = np.array([pdfs[galname]['vesc']
                         for galname in galnames]).flatten()

    ###########################################################################
    ## Fitting 
    ###########################################################################
    def calc_p(vs, vcircs, vescs, d, e, p):
        '''
        Calculate probability density given velocity, circular velocity, and
        escape velocity
        '''
        ps = mao(
            vs,
            d * (vcircs / 100.) ** e,
            vescs,
            p
        )
        return ps
    
    model = lmfit.model.Model(calc_p,
                              independent_vars=['vs', 'vcircs', 'vescs'])
    params = model.make_params()

    params['d'].set(value=50.770942240409525, vary=True, min=0.)
    params['e'].set(value=2.4522990491102967, vary=True, min=0.)
    params['p'].set(value=2.3049471166082682, vary=True, min=0.001, max=15.)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 
                                category=scipy.integrate.IntegrationWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        result = model.fit(ps_truth, params, 
                           vs=vs_truth, 
                           vcircs=vcircs, 
                           vescs=vescs,
                           method='nelder')

    if update_values:
        # Save raw variables to data_raw.pkl
        data2save = {key: result.params[key].value
                     for key in result.params.keys()}
        stderrs = {key+'_stderr': result.params[key].stderr
                   for key in result.params.keys()}
        covar = {'covar': result.covar}
        data2save = data2save | stderrs | covar #combine dictionaries
        dm_den.save_var_raw(data2save, 'results_mao_' + vcut_type + '.pkl')

        # Save to the LaTeX data file
        for key in result.params.keys():
            if result.params[key].vary: 
                # Save strings to be used in paper.tex
                y = result.params[key].value
                stderr = result.params[key].stderr
                ddfrac, dpfrac = grid_eval_mao.identify('grid_mao.h5')
                if key == 'd':
                    dy = ddfrac * y
                elif key == 'p':
                    dy = dpfrac * y
                else:
                    #Number of parameters we're estimating
                    p = result.covar.shape[0] 

                    N = 600 #There's 600 data points (50 bins for each disc) 
                    z = 1. #Number of std deviations in forecast_sig
                    P = scipy.special.erf(z / np.sqrt(2)) 
                    forecast_sig = 1. - P

                    # critical 2-tailed t value  
                    tc = scipy.stats.t.ppf(q=1.-forecast_sig/2., df=N-p)

                    dy = stderr * tc
                # y_txt is a string. DY_TXT is an array of strings, or just an
                # array of just one string. Either way, 
                # type(DY_TXT) == np.ndarray, which is why we're denoting it in
                # all caps.
                y_txt, DY_TXT = staudt_utils.sig_figs(y, dy)
                uci.save_prediction(key + '_mao_' + vcut_type, 
                                       y_txt,  DY_TXT)
        
    ###########################################################################
    return result

def fit_mao_naive_aggp(
        vcut_type, 
        df_source, 
        raw_results_fname=None):
    '''
    Find the best p parameter based on the aggregation of all twelve discs.
    v0 is assumed to be the circular speed.
    
    Parameters
    ----------
    vcut_type: {'lim_fit', 'lim', 'vesc_fit', 'vesc', 'ideal'}
        Specifies how to determine the speed distribution cutoff.
    df_source: str
        The name of the file containing the analysis DataFrame
    raw_results_fname: str, default None
        The name of the file where the raw, unrounded, float results should be
        saved.

    Returns
    -------
    result: lmfit.ModelResult
        The result of the fit
    '''
    if vcut_type != 'lim_fit' and raw_results_fname is not None:
        raise ValueError('You should only update values if you\'re using'
                         ' vcut_type=\'lim_fit\'.')
    import dm_den
    import dm_den_viz
    df = dm_den.load_data(df_source).drop(['m12w', 'm12z'])
    pdfs = copy.deepcopy(pdfs_v)
    galnames = list(pdfs.keys())
    vcut_dict = dm_den.load_vcuts(vcut_type, df)
    for gal in ['m12z', 'm12w']:
        galnames.remove(gal)
    for gal in galnames:
        dict_gal = pdfs[gal]
        bins = dict_gal['bins']
        dict_gal['vcirc'] = np.repeat(
            df.loc[gal, 'v_dot_phihat_disc(T<=1e4)'],
            len(dict_gal['ps']))
        dict_gal['vesc'] = np.repeat(vcut_dict[gal],
                                     len(dict_gal['ps']))

    ps_truth = np.array([pdfs[galname]['ps']
                   for galname in galnames]).flatten()
    vs_truth = np.array([pdfs[galname]['vs']
                   for galname in galnames]).flatten()
    
    N_postfit = 300
    vs_postfit = np.linspace(0., 700., N_postfit)

    vcircs = np.array([pdfs[galname]['vcirc']
                         for galname in galnames]).flatten()
    vescs = np.array([pdfs[galname]['vesc']
                         for galname in galnames]).flatten()

    ###########################################################################
    ## Fitting 
    ###########################################################################
    def calc_p(vs, vcircs, vescs, p):
        '''
        Calculate probability density given velocity, circular velocity, and
        escape velocity
        '''
        ps = mao(
            vs,
            vcircs,                  
            vescs,
            p
        )
        return ps
    
    model = lmfit.model.Model(calc_p,
                              independent_vars=['vs', 'vcircs', 'vescs'])
    params = model.make_params()

    params['p'].set(value=2.5, vary=True, min=0.1, max=10.)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 
                                category=scipy.integrate.IntegrationWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        result = model.fit(ps_truth, params, 
                           vs=vs_truth, 
                           vcircs=vcircs, 
                           vescs=vescs,
                           method='nelder')
    ###########################################################################

    if raw_results_fname is not None and vcut_type == 'lim_fit':
        # Save the best p value to our big raw data file.
        y = result.params['p'].value
        dm_den.save_var_raw({'p_mao_naive_agg': y},
                            raw_results_fname)
        # Also update the paper
        # For now, we're just going to save p with 1 decimal showing to match
        # the 1 decimal that shows when we carry out the full error analysis on
        # our full version of the Mao model
        uci.save_var_latex('p_mao_naive_agg', '{0:0.1f}'.format(y))

    return result

def fit_mao_naive_indvp(gal):
    '''
    Find the best p parameter for each individual galaxy.
    v0 is assumed to be the circular speed.
    vesc is *predicted* from fitting veschat(vc) = vesc0 * (vc/100) ^ epsilon
    '''
    import dm_den
    import dm_den_viz

    df = dm_den.load_data('dm_stats_dz1.0_20230626.h5')
    vcut_hat_dict = dm_den.load_vcuts('lim_fit', df)

    ###########################################################################
    ## Fitting 
    ###########################################################################
    def calc_p(vs, vc, vesc, p):
        '''
        Calculate probability density given velocity, circular velocity, and
        escape velocity
        '''
        ps = mao(
            vs,
            vc,
            vesc,
            p
        )
        return ps
    
    model = lmfit.model.Model(calc_p,
                              independent_vars=['vs', 'vc', 'vesc'])
    params = model.make_params()
    params['p'].set(value=2.5, vary=True, min=0.001, max=10.)

    ps_truth = pdfs_v[gal]['ps']
    vs_truth = pdfs_v[gal]['vs']
    vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
    vesc = vcut_hat_dict[gal]
    with warnings.catch_warnings():
        warnings.filterwarnings(
                'ignore', 
                category=scipy.integrate.IntegrationWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        result = model.fit(ps_truth, params, 
                           vs=vs_truth, 
                           vc=vc, 
                           vesc=vesc,
                           method='nelder')
    return result.params['p'].value

def fit_universal_sigmoid_exp(update_values=False,
                              method='leastsq', 
                              vc100=True, **kwargs):
    import dm_den
    import dm_den_viz
    df = dm_den.load_data('dm_stats_dz1.0_20230626.h5').drop(['m12w', 'm12z'])
    pdfs = copy.deepcopy(pdfs_v)
    pdfs.pop('m12z')
    pdfs.pop('m12w')
    vlim_fit_dict = dm_den.load_vcuts('lim_fit', df)
    for gal in pdfs:
        dict_gal = pdfs[gal]
        bins = dict_gal['bins']
        dict_gal['vcirc'] = np.repeat(
            df.loc[gal, 'v_dot_phihat_disc(T<=1e4)'],
            len(dict_gal['ps']))
        #dict_gal['vesc'] = np.repeat(vesc_dict[gal]['ve_avg'],
        #                             len(dict_gal['ps']))
        dict_gal['vlim_fit'] = np.repeat(vlim_fit_dict[gal],
                                         len(dict_gal['ps']))

    ps_truth = np.array([pdfs[galname]['ps']
                   for galname in pdfs]).flatten()
    vs_truth = np.array([pdfs[galname]['vs']
                   for galname in pdfs]).flatten()
    
    N_postfit = 300
    vs_postfit = np.linspace(0., 700., N_postfit)

    vcircs = np.array([pdfs[galname]['vcirc']
                         for galname in pdfs]).flatten()
    #vescs = np.array([pdfs[galname]['vesc']
    #                     for galname in pdfs]).flatten()
    vlim_fits = np.concatenate([pdfs[gal]['vlim_fit'] for gal in pdfs])

    ###########################################################################
    ## Fitting 
    ###########################################################################
    def calc_p(vs, vcircs, vlim_fits, d, e, h, j, k):
        '''
        Calculate probability density given velocity and circular velocity
        '''
        if vc100:
            vcircs = vcircs.copy()
            vcircs /= 100.
        ps = [max_double_exp(v,
                             d * (vc) ** e,
                             h * (vc) ** j,
                             k,
                             vlim_fit)
              for v, vc, vlim_fit in zip(vs, vcircs, vlim_fits)]
        ps = np.array(ps)
        return ps
    
    model = lmfit.model.Model(calc_p,
                              independent_vars=['vs', 'vcircs', 'vlim_fits'])
    params = model.make_params()

    if vc100:
        params['d'].set(value=114.970072, vary=True, min=0.)
        params['e'].set(value=0.92818194, vary=True, min=0.)
        params['h'].set(value=388.227498, vary=True, min=0.)
        params['j'].set(value=0.27035486, vary=True, min=0.)
        params['k'].set(value=0.03089876, vary=True, min=0.0001, max=1.)
    else:
        params['d'].set(value=1.60030613, vary=True, min=0.1, max=4.)
        params['e'].set(value=0.92819047, vary=True, min=0.1, max=4.)
        params['h'].set(value=111.783463, vary=True, min=5., max=300.)
        params['j'].set(value=0.27035357, vary=True, min=0.05, max=4.)
        params['k'].set(value=0.03089876, vary=False, min=0.0001, max=1.)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 
                                category=scipy.integrate.IntegrationWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        result = model.fit(ps_truth, params, vs=vs_truth, 
                           vcircs=vcircs, vlim_fits=vlim_fits,
                           method=method, **kwargs)

    if update_values:
        # Save raw variables to data_raw.pkl
        data2save = {key: result.params[key].value
                     for key in result.params.keys()}
        stderrs = {key+'_stderr': result.params[key].stderr
                   for key in result.params.keys()}
        covar = {'covar': result.covar}
        data2save = data2save | stderrs | covar #combine dictionaries
        with open(paths.data + 'params_sigmoid_exp.pkl', 'wb') as f:
            pickle.dump(data2save, f, pickle.HIGHEST_PROTOCOL)
    ###########################################################################

    return result

def plt_universal(gals='discs', update_values=False,
                  tgt_fname=None, method='leastsq', 
                  vc100=True, err_method='sampling', ddfrac=None, dhfrac=None,
                  assume_corr=False,
                  band_alpha=0.9, data_color='k', 
                  band_color=plt.cm.viridis(1.),
                  samples_color=plt.cm.viridis(0.5), ymax=None, show_rms=False,
                  pdfs_fname='v_pdfs_disc_dz1.0_20240606.pkl',
                  raw_results_fname=None,
                  **kwargs):
    '''
    Noteworthy Parameters
    ---------------------
    update_values: bool, default False
        Whether to update the raw data results, paper data, rms raw data, and
        saved lmfit.model instance. If True, the user can also specify a
        `raw_results_fname` where the raw data results will be stored. If the
        user doesn't specify a `raw_results_fname` but does set `update_values`
        to True, raw data results will be put in data_raw.pkl.
    err_method: {'sampling', 'std_err', None}, default 'sampling'
        The method to use to generate the error bands
    pdfs_fname: str
        The filename from which to get the galaxies' true probability densities
    raw_results_fname: str, default None
        Filename to which the program will save its raw data results in float
        format with no rounding
    '''
    if err_method not in ['sampling', 'std_err', None]:
        raise ValueError('Unexpected argument for `err_method`.')
    if (ddfrac is not None or dhfrac is not None or assume_corr) \
        and err_method != 'sampling':
        raise ValueError('ddfrac, dhfrac, and assume_corr are only used in '
                         'sampling.')
    elif err_method == 'sampling':
        grid_results = grid_eval.identify()
        if ddfrac is None:
            ddfrac = grid_results[0]
            print('Using ddfrac = {0:0.5f}'.format(ddfrac))
        if dhfrac is None:
            dhfrac = grid_results[1]
            print('Using dhfrac = {0:0.5f}'.format(dhfrac))
    if update_values and gals != 'discs':
        raise ValueError('You should only update values when you\'re plotting '
                         'all the discs.')
    if not update_values and raw_results_fname is not None:
        raise ValueError('Only specify the raw_results_fname if you are'
                         'updating values.')

    import dm_den
    import dm_den_viz
    df = dm_den.load_data('dm_stats_dz1.0_20230626.h5').drop(['m12w', 'm12z'])
    if gals != 'discs' and not isinstance(gals, (list, np.ndarray)):
        raise ValueError('Unexpected value provided for gals arg')
    if gals == 'discs':
        Ngals = 12
    else:
        Ngals = len(gals)
    with open(paths.data + pdfs_fname, 'rb') as f:
        pdfs = pickle.load(f)
    pdfs.pop('m12z')
    pdfs.pop('m12w')
    for gal in pdfs:
        dict_gal = pdfs[gal]
        bins = dict_gal['bins']
        dict_gal['vcirc'] = np.repeat(
            df.loc[gal, 'v_dot_phihat_disc(T<=1e4)'],
            len(dict_gal['ps']))
        dict_gal['vesc'] = np.repeat(vesc_dict[gal]['ve_avg'],
                                     len(dict_gal['ps']))

    ps_truth = np.array([pdfs[galname]['ps']
                   for galname in pdfs]).flatten()
    vs_truth = np.array([pdfs[galname]['vs']
                   for galname in pdfs]).flatten()
    
    N_postfit = 300
    vs_postfit = np.linspace(0., 700., N_postfit)

    vcircs = np.array([pdfs[galname]['vcirc']
                         for galname in pdfs]).flatten()
    vescs = np.array([pdfs[galname]['vesc']
                         for galname in pdfs]).flatten()

    ###########################################################################
    ## Fitting 
    ###########################################################################
    def calc_p(vs, vcircs, d, e, h, j, k):
        '''
        Calculate probability density given velocity and circular velocity
        '''
        if vc100:
            vcircs = vcircs.copy()
            vcircs /= 100.
        ps = smooth_step_max(vs,
                             d * (vcircs) ** e,
                             h * (vcircs) ** j,
                             k)
        return ps
    
    model = lmfit.model.Model(calc_p,
                              independent_vars=['vs', 'vcircs'])
    params = model.make_params()

    if vc100:
        params['d'].set(value=114.970072, vary=True, min=0.)
        params['e'].set(value=0.92818194, vary=True, min=0.)
        params['h'].set(value=388.227498, vary=True, min=0.)
        params['j'].set(value=0.27035486, vary=True, min=0.)
        params['k'].set(value=0.03089876, vary=True, min=0.0001, max=1.)
    else:
        params['d'].set(value=1.60030613, vary=True, min=0.1, max=4.)
        params['e'].set(value=0.92819047, vary=True, min=0.1, max=4.)
        params['h'].set(value=111.783463, vary=True, min=5., max=300.)
        params['j'].set(value=0.27035357, vary=True, min=0.05, max=4.)
        params['k'].set(value=0.03089876, vary=False, min=0.0001, max=1.)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 
                                category=scipy.integrate.IntegrationWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        result = model.fit(ps_truth, params, vs=vs_truth, 
                           vcircs=vcircs, method=method, **kwargs)

    if update_values:
        # Save ModelResult object
        lmfit.model.save_modelresult(
                result, 
                paths.data + 'sigmoid_damped_ls_result.sav'
        )

        # Save raw variables to `raw_results_fname`
        data2save = {key: result.params[key].value
                     for key in result.params.keys()}
        stderrs = {key+'_stderr': result.params[key].stderr
                   for key in result.params.keys()}
        covar = {'covar': result.covar}
        data2save = data2save | stderrs | covar #combine dictionaries
        dm_den.save_var_raw(data2save, raw_results_fname)

        p = result.covar.shape[0] #Number of parameters we're estimating
        N = 600 #There's 600 data points (50 bins for each disc) 
        z = 1. #Number of std deviations in forecast_sig
        #Probability of being within z std devs:
        P = scipy.special.erf(z / np.sqrt(2)) 
        forecast_sig = 1. - P

        # critical 2-tailed t value  
        tc = scipy.stats.t.ppf(q=1.-forecast_sig/2., df=N-p)

        if update_paper:
            for key in result.params.keys():
                if result.params[key].vary: 
                    # Save strings to be used in paper.tex
                    y = result.params[key].value
                    stderr = result.params[key].stderr
                    if key == 'd' and err_method is not None:
                        dy = ddfrac * y
                    elif key == 'h' and err_method is not None:
                        dy = dhfrac * y
                    else:
                        dy = stderr * tc
                    # y_txt is a string. DY_TXT is an array of strings, or just 
                    # an
                    # array of just one string. Either way, 
                    # type(DY_TXT) == np.ndarray, which is why we're denoting 
                    # it in
                    # all caps.
                    y_txt, DY_TXT = staudt_utils.sig_figs(y, dy)
                    uci.save_prediction(key, y_txt,  DY_TXT)
            vc_mw = dm_den_viz.vc_eilers 

            v0_mw = result.params['d'] * (vc_mw / 100.) ** result.params['e']
            dv0_mw = ddfrac * v0_mw
            v0_mw_txt, DV0_MW_TXT = staudt_utils.sig_figs(v0_mw, dv0_mw)
            uci.save_prediction('v0_mw', v0_mw_txt, DV0_MW_TXT)

            vdamp_mw = (
                result.params['h'] * (vc_mw / 100.) ** result.params['j']
            )
            dvdamp_mw = dhfrac * vdamp_mw
            vdamp_mw_txt, DVDAMP_MW_TXT = staudt_utils.sig_figs(vdamp_mw, 
                                                                dvdamp_mw)
            # Save to LaTeX paper.
            uci.save_prediction('vdamp_mw', vdamp_mw_txt, DVDAMP_MW_TXT)
    ###########################################################################

    if err_method == 'std_err':
        # Calculate the std error of the regression
        # p = 4  degrees of freedom negated by estimating d, e, h, and j:
        p = result.nvarys 
        # Std err of the regression
        s = np.sqrt(np.sum(result.residual ** 2.) / (result.ndata - p)) 

        N_vc = len(pdfs)
        vc_mean = df['v_dot_phihat_disc(T<=1e4)'].mean()
        # Variance in vc:
        var_vc = np.sum((df['v_dot_phihat_disc(T<=1e4)'] \
                            - vc_mean)**2.) / N_vc
        # Variance in v
        var_vs = np.sum((vs_truth.flatten() - vs_truth.mean())**2.) \
                 / result.ndata

    fig, axs = dm_den_viz.setup_multigal_fig(gals)

    if isinstance(gals, (list, np.ndarray)):
        df = df.loc[gals]

    rms_dict = {}
    if show_rms:
        sse = 0.
        N_data = 0.
    pbar = ProgressBar()
    for i, gal in enumerate(pbar(df.index)):
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        vcircs_postfit = np.repeat(vc, N_postfit)
        ps_postfit = result.eval(vs=vs_postfit, vcircs=vcircs_postfit)

        vs_truth = pdfs[gal]['vs']
        rms_err, sse_add = calc_rms_err(
                vs_truth, pdfs[gal]['ps'], 
                calc_p,  
                [np.repeat(vc, len(pdfs[gal]['vs'])), 
                *[result.params[key] for key in ['d', 'e', 
                                                 'h', 'j', 
                                                 'k']]]
        )
        if show_rms:
            sse += sse_add
            N_data += len(vs_truth)
        rms_dict[gal] = rms_err
        if err_method == 'std_err':
            # Std err of the vcirc mean at vc_gal
            #     (Dividing by sqrt(N_vc) and not sqrt(N) is my way of trying 
            #     to impart
            #     into the uncertainty calculation
            #     the fact that I only have 12 galaxies.)
            s_vc = s / np.sqrt(N_vc) \
                   * np.sqrt(1. + (vc - vc_mean)**2. / var_vc)
            # Std err of the vs_truth mean at each dict_gal['vs']
            s_vs = s / np.sqrt(result.ndata) \
                   * np.sqrt(1. + (vs_postfit - vs_truth.mean())**2. / var_vs)
            # Std err of the predictions
            s_prediction = np.sqrt(s**2. + s_vc**2. + s_vs**2.)

            axs[i].fill_between(vs_postfit,
                                ps_postfit - s_prediction,
                                ps_postfit + s_prediction,
                                color=band_color, alpha=0.4, ec=None,
                                label='$1\sigma$ band')
        elif err_method == 'sampling':
            lowers, uppers = gal_bands(gal, vs_postfit, df, 
                                       result, ddfrac=ddfrac, dhfrac=dhfrac,
                                       assume_corr=assume_corr,
                                       ax = axs[i], 
                                       samples_color=samples_color)

            axs[i].fill_between(vs_postfit, lowers, uppers, color=band_color, 
                                alpha=band_alpha, 
                                ec=samples_color, zorder=1, 
                                label='$1\sigma$ band')
        # Plot data
        axs[i].stairs(pdfs[gal]['ps'], pdfs[gal]['bins'], color=data_color,
                      label='data')
        # Plot prediction
        axs[i].plot(vs_postfit,
                    ps_postfit,
                    '-',
                    label='prediction from $v_\mathrm{c}$', color='C3', lw=1.5)
        # Make ticks on both sides of the x-axis:
        axs[i].tick_params(axis='x', direction='inout', length=6)
        # Put y-axis in scientific notation
        order_of_mag = -3
        axs[i].ticklabel_format(style='sci', axis='y', 
                                scilimits=(order_of_mag,
                                           order_of_mag),
                                useMathText=True)
        if Ngals == 2:
            # Remove the 0 tick label because of overlap
            y0, y1 = axs[i].get_ylim()
            visible_ticks = np.array([t for t in axs[i].get_yticks() \
                                      if t>=y0 and t<=y1])
            new_ticks = visible_ticks[visible_ticks > 0.]
            axs[i].set_yticks(new_ticks)

            # Draw residual plot
            vs_resids = copy.deepcopy(pdfs[gal]['vs'])
            vs_extend = np.linspace(vs_resids.max(), vs_postfit.max(), 20)
            vs_resids = np.append(vs_resids, vs_extend, axis=0)                                         
            
            ps_hat = result.eval(vs=pdfs[gal]['vs'], 
                                 vcircs=np.repeat(vc, len(pdfs[gal]['vs'])))
            resids = ps_hat - pdfs[gal]['ps']
            resids_extend = result.eval(vs=vs_extend,
                                        vcircs=np.repeat(vc, len(vs_extend)))
            resids = np.append(resids, resids_extend, axis=0)
            axs[i+2].plot(vs_resids, resids / 10.**order_of_mag, color='C3')
            axs[i+2].axhline(0., linestyle='--', color='k', alpha=0.5, lw=1.)
            axs[i+2].set_ylim(-dm_den_viz.resids_lim, dm_den_viz.resids_lim)
            if i == 0:
                axs[i+2].set_ylabel('resids')
        loc = [0.97,0.96]
        kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                          va='top', ha='right',
                          bbox=dict(facecolor='white', alpha=0.8, 
                                    edgecolor='none'))
        axs[i].annotate(gal, loc,
                        **kwargs_txt)
        loc[1] -= 0.15
        kwargs_txt['fontsize'] = 11.
        rms_txt = staudt_utils.mprint(rms_err, d=1, show=False).replace('$','')
        axs[i].annotate('$v_\mathrm{{c}}={0:0.0f}\,\mathrm{{km\,s^{{-1}}}}$'
                        #'\n$\mathrm{{RMS}}_\mathrm{{err}}={1:s}$'
                        .format(vc, 
                                rms_txt
                               ),
                        loc, **kwargs_txt)
        axs[i].grid(False)
        if ymax is not None:
            axs[i].set_ylim(top=ymax)

    dm_den_viz.label_axes(axs, fig, gals)
    if fig.Nrows == 3:
        legend_y = 0.
        ncol = 4
    else:
        legend_y = -0.04
        ncol = 2
    handles, labels = axs[0].get_legend_handles_labels()
    handles.append(mpl.lines.Line2D([0], [0], color=samples_color, lw=1.,
                                    label='rand samples'))
    axs[0].legend(handles=handles,
                  bbox_to_anchor=(0.5, legend_y), 
                  loc='upper center', ncol=ncol,
                  bbox_transform=fig.transFigure)

    plt.draw()
    for i in range(len(df.index)):
        # Need to draw everything before going back and doing this because
        # otherwise we'll lock in the upper y limit (x limit) for each row with
        # the
        # automatically determined upper limit for the galaxy in the first
        # column (row) of that row (column).
        axs[i].set_ylim(bottom=0.)
        axs[i].set_xlim(left=0.)
    plt.show()

    if show_rms:
        d = 2
        txt = staudt_utils.mprint(np.sqrt(sse / N_data),
                                  d=d,
                                  show=False).replace('$', '')
        display(Latex('$\mathrm{{RMS}}={0:s}$'.format(txt)))

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=140)

    return result

def plt_universal_mc(gals='discs', m=1.):
    '''
    Plot uncertainty in distribution with a monte carlo method

    Parameters
    ----------
    gals: str or list-like of str
        Which galaxies to plot. The fitting method uses all discs regardless of
        how this parameter is set.
    m: float
        A multiplier applied to the standard error on each parameter in the
        monte carlo

    Returns
    -------
    result: lmfit.minimizer.MinimizerResult
        Result of the universal fit
    '''
    import dm_den

    df = dm_den.load_data('dm_stats_dz1.0_20230626.h5')
    df.drop(['m12z', 'm12w'], inplace=True)

    pdfs = copy.deepcopy(pdfs_v)

    vary_dict = {'d': True,
                 'e': True,
                 'h': True,
                 'j': False,
                 'k': False}
    vary_mask = np.array(list(vary_dict.values()))
    result = fit_universal_no_uncert(vary_dict=vary_dict, plot=False, 
                                     update_vals=False)
    params = result.params
    covar = result.covar

    p = covar.shape[0] #Number of parameters we're estimating
    N = 600 #There's 600 data points (50 bins for each disc) 
    z = 1. # Number of std deviations to include in forecast_sig
    #Probability of being within z std devs:
    P = scipy.special.erf(z / np.sqrt(2)) 
    forecast_sig = 1. - P

    # critical 2-tailed t value  
    tc = scipy.stats.t.ppf(q=1.-forecast_sig/2., df=N-p)
    tc *= m 

    vs = np.linspace(0., 675., 1000)

    d_best = params['d'].value
    e_best = params['e'].value
    h_best = params['h'].value
    j_best = params['j'].value
    k = params['k'].value

    # c is a matrix for which c*c^T = covar
    c = scipy.linalg.cholesky(covar, lower=True)

    N = 1000
    d_dev_samples = np.random.normal(0., result.params['d'].stderr * tc, 
                                     size=N)
    e_dev_samples = np.random.normal(0., result.params['e'].stderr * tc, 
                                     size=N)
    h_dev_samples = np.random.normal(0., result.params['h'].stderr * tc, 
                                     size=N)
    j_dev_samples = np.random.normal(0., result.params['j'].stderr * tc, 
                                     size=N)
    k_dev_samples = np.random.normal(0., result.params['k'].stderr * tc, 
                                     size=N)

    assert list(vary_dict.keys()) == ['d', 'e', 'h', 'j', 'k']
    mu_theta = np.array([d_best, e_best, h_best, j_best, k])
    mu_theta = mu_theta[vary_mask] #Only use parameters we're varying
    mu_theta = mu_theta.reshape((p, 1))
    # Matrix of uncorrelated deviations from the best estimates:
    Devs_uncorr = np.array([d_dev_samples, e_dev_samples, h_dev_samples, 
                            j_dev_samples, k_dev_samples])
    Devs_uncorr = Devs_uncorr[vary_mask] #Only use parameters we're varying
    # Apply correlation to parameter samples
    theta = np.dot(c, Devs_uncorr) + mu_theta 

    # Set parameters to either samples or their best fit value for later 
    # calculations
    i = 0
    if vary_dict['d']:
        d = theta[i]
        i += 1
    else:
        d = d_best
    if vary_dict['e']:
        e = theta[i]
        i += 1
    else:
        e = e_best
    if vary_dict['h']:
        h = theta[i]
        i += 1
    else:
        h = h_best
    if vary_dict['j']:
        j = theta[i]
    else:
        j = j_best

    ###########################################################################
    # Plot parameters
    ###########################################################################
    fig, axs = plt.subplots(p, p, dpi=120., sharex='col', sharey='row')
    fig.subplots_adjust(wspace=0., hspace=0.)
    for i in range(p):
        for l in range(p):
            if i <= l:
                axs[i,l].remove()
                continue
            axs[i, l].plot(theta[l], theta[i], 'o', ms=.3)
            axs[i, l].grid(False)
            if l == 0:
                axs[i, l].set_ylabel('$\\theta_{{{0:0.0f}}}$'.format(i),
                                     fontsize=10.)
            axs[i, l].set_xlabel('$\\theta_{{{0:0.0f}}}$'.format(l),
                                 fontsize=10.)
            axs[i, l].tick_params(axis='both', labelsize=10.)
    plt.show()
    ###########################################################################
            
    fig, axs = dm_den_viz.setup_multigal_fig(gals)
    
    if isinstance(gals, (list, np.ndarray)):
        df = df.loc[gals]

    resids = {}
    s_dict = {}

    for i, gal in enumerate(df.index):
        ax = axs[i]
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']

        v0 = d * vc ** e 
        vdamp = h * vc ** j 

        bins = pdfs[gal]['bins']
        vs_truth = pdfs[gal]['vs']
        Ps_hat = np.array([smooth_step_max(vs_truth, v0_, vdamp_, k) 
                           for v0_, vdamp_ in zip(v0, vdamp)])
        ps_truth = pdfs[gal]['ps']
        resids_gal = Ps_hat - ps_truth
        resids[gal] = resids_gal 
 
        s_dict[gal] = np.sqrt( np.sum( resids_gal.flatten() ** 2. ) 
                               / ( len(resids_gal.flatten()) - 4) )

        for l in range(N): 
            #ax.plot(vs, smooth_step_max(vs, v0[l], vdamp[l], k), 
            #        c='b', alpha=0.05)
            ax.plot(vs_truth, Ps_hat[l],
                    c='b', alpha=0.05)

        ax.stairs(pdfs[gal]['ps'], pdfs[gal]['bins'], color='grey')
        ax.plot(vs, smooth_step_max(vs, d_best * vc ** e_best,
                                    h_best * vc ** j_best,
                                    k), 
                c='r', lw=1.)

        loc = [0.97,0.96]
        kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                          va='top', ha='right',
                          bbox=dict(facecolor='white', alpha=0.8, 
                                    edgecolor='none'))
        axs[i].annotate(gal, loc,
                        **kwargs_txt)

        axs[i].grid(False)
    dm_den_viz.label_axes(axs, fig, gals=gals)

    plt.show()

    return result

###############################################################################

def calc_rms_err(xs, ys, fcn, args):
    ys_hat = fcn(xs, *args)
    N = len(ys_hat)
    sse = np.sum((ys_hat - ys)**2.)
    rms = np.sqrt( sse / N )
    return rms, sse 

def calc_rms_all_methods(
        df_source,
        pdfs_fname='v_pdfs_disc_dz1.0_20240606.pkl',
        staudt_results_fname='results_mcmc.pkl',
        mao_naive_aggp_results_fname='data_raw.pkl',
        update_paper=False):
    import dm_den
    with open(paths.data + pdfs_fname, 'rb') as f:
        pdfs = pickle.load(f)
    with open(paths.data + mao_naive_aggp_results_fname, 'rb') as f:
        p_naive = pickle.load(f)['p_mao_naive_agg']
    with open(paths.data + staudt_results_fname, 'rb') as f:
        staudt_results = pickle.load(f)
    with open(paths.data + 'results_mao_lim_fit.pkl', 'rb') as f:
        mao_ours_results = pickle.load(f)
    df = dm_den.load_data(df_source)
    vesc_dict = dm_den.load_vcuts('lim_fit', df)

    galnames = df.drop(['m12z', 'm12w']).index
    
    for galname in galnames:
        vc = df.loc[galname, 'v_dot_phihat_disc(T<=1e4)']
        vesc = vesc_dict[galname]
        N = len(pdfs[galname]['vs'])
        pdfs[galname]['vc'] = np.repeat(vc, N)
        pdfs[galname]['vesc'] = np.repeat(vesc, N)
    vs = np.concatenate([pdfs[galname]['vs'] for galname in galnames])
    ys = np.concatenate([pdfs[galname]['ps'] for galname in galnames])
    vcs = np.concatenate([pdfs[galname]['vc'] for galname in galnames])
    vescs = np.concatenate([pdfs[galname]['vesc'] for galname in galnames])
    
    mao_naive_rms = calc_rms_err(vs, ys, mao, (vcs, vescs, p_naive))[0]

    v0s_mao = mao_ours_results['d'] * (vcs / 100.) ** mao_ours_results['e']
    mao_ours_rms = calc_rms_err(
        vs, 
        ys, 
        mao, 
        (v0s_mao, vescs, mao_ours_results['p'])
    )[0]

    v0s_staudt = staudt_results['d'] * (vcs / 100.) ** staudt_results['e']
    vdamps = staudt_results['h'] * (vcs / 100.) ** staudt_results['j']
    staudt_rms = calc_rms_err(
        vs, 
        ys,
        smooth_step_max,
        (v0s_staudt, vdamps, staudt_results['k'])
    )[0]

    if update_paper:
        uci.save_var_latex(
            'staudt_rms',
            staudt_utils.mprint(
                staudt_rms,
                d=2,
                show=False,
                order_of_mag=-3).replace('$', '')
        )
        uci.save_var_latex(
            'mao_naive_rms',
            staudt_utils.mprint(
                mao_naive_rms,
                d=2,
                show=False,
                order_of_mag=-3).replace('$', '')
        )
        uci.save_var_latex(
            'mao_ours_rms',
            staudt_utils.mprint(
                mao_ours_rms,
                d=2,
                show=False,
                order_of_mag=-3).replace('$', '')
        )
        
    return staudt_rms, mao_naive_rms, mao_ours_rms

def extend(pdfs, df, result):
    '''
    Extend the given pdf dictionary in place.

    Parameters
    ----------
    pdfs: dict
        Dictionary of the the velocity probability density data
    df: pandas.DataFrame
        Dataframe containing analysis results
    result: dict
        Dictionary of the universal fit result

    Returns
    -------
    None
    '''
    for gal in pdfs:
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        dict_gal = pdfs[gal]
        bins = dict_gal['bins']

        bin_spaces = bins[1:] - bins[:-1]
        bin_spacing = np.mean(bin_spaces)
        # bins_extend[0] == dict_gal['bins'][-1]. This is good for
        # generating vs_extend, but bins_extend[0] must be excluded later
        # when appending bins_extend to dict_gal['bins'].
        bins_extend = np.arange(bins.max(),
                                700. + bin_spacing,
                                bin_spacing)
        vs_extend = (bins_extend[1:] + bins_extend[:-1]) / 2.
        ps_extend = np.zeros(len(vs_extend))
        
        # Check where the extension becomes counterproductive
        ps_hat_extend = [smooth_step_max(
                              v, 
                              result['d'] * (vc/100.) ** result['e'],
                              result['h'] * (vc/100.) ** result['j'],
                              result['k'])
                         for v in vs_extend]
        ps_hat_extend = np.array(ps_hat_extend)

        keep = ps_hat_extend >= 1e-5
        # The False concatenation serves to remove bins_extend[0], the
        # necessity of which was mentioned above.
        bins_extend = bins_extend[np.concatenate([[False], keep], axis=0)]
        vs_extend = vs_extend[keep]
        ps_extend = ps_extend[keep]
        ps_hat_extend = ps_hat_extend[keep]

        bins = np.append(bins, bins_extend, axis=0)
        dict_gal['bins'] = bins
        dict_gal['ps'] = np.append(dict_gal['ps'], ps_extend)

        dict_gal['vs'] = (bins[:-1] + bins[1:]) / 2.
        dict_gal['vcirc'] = np.repeat(
            vc, 
            len(dict_gal['ps']))
    return None

def find_uncertainty(gals, ddfrac=0.1, dhfrac=0.18, v0char=1., N_samples=1000,
                     assume_corr=False):
    '''
    Parameters
    ----------
    gals: str or list-like of str
        Which galaxies to plot
    ddfrac: float
        Fractional uncertainty on d parameter
    dhfrac: float
        Fractional uncertainty on h parameter
    v0char: float
        Characteristic peak velocity to use
    N_samples: int
        Number of times to sample d and h when making the uncertainty bands

    Returns
    -------
    None

    '''
    import dm_den

    if gals != 'discs' and not isinstance(gals, (list, np.ndarray)):
        raise ValueError('Unexpected value provided for gals arg')
    
    df = dm_den.load_data('dm_stats_dz1.0_20230626.h5').drop(['m12w', 'm12z'])
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        result = pickle.load(f)
    pdfs = copy.deepcopy(pdfs_v)
    del pdfs['m12w']
    del pdfs['m12z']

    extend(pdfs, df, result)

    vs_truth = np.concatenate([pdfs[galname]['vs']
                   for galname in pdfs])
    ps_truth = np.concatenate([pdfs[galname]['ps']
                   for galname in pdfs])
    vcircs = np.concatenate([pdfs[galname]['vcirc']
                         for galname in pdfs])
    ps_hat = [smooth_step_max(v, 
                              result['d'] * (vc/100.) ** result['e'],
                              result['h'] * (vc/100.) ** result['j'],
                              result['k'])
              for v, vc in zip(vs_truth, vcircs)]
    ps_hat = np.array(ps_hat)
    resids = ps_hat - ps_truth

    N = len(resids)
    p = 4 #4  degrees of freedom negated by estimating d, e, h, and j
    s = np.sqrt(np.sum(resids ** 2.) / (N - p)) #std err of the regression

    N_vc = len(pdfs)
    vc_mean = df['v_dot_phihat_disc(T<=1e4)'].mean()
    var_vc = np.sum((df['v_dot_phihat_disc(T<=1e4)'] \
                        - vc_mean)**2.) / N_vc
    var_vs = np.sum((vs_truth.flatten() - vs_truth.mean())**2.) / N

    ###########################################################################
    # Plot standardized galaxies 
    ###########################################################################
    def get_std_vdamp_k(vc):
        def calc_p(vs, d, h, k):
            '''
            Calculate probability density given velocity and circular velocity
            '''
            ps = np.array([smooth_step_max(v,
                                           d * (vc/100.) ** e,
                                           h * (vc/100.) ** j,
                                           k)
                           for v in vs])
            return ps
        # Dividing and multiplying by each galaxy's PREDICTED V0  puts the 
        # peaks
        # of true distributions around 1.
        # Then multiplying and dividing by v0char puts the peaks around v0char.
        # The default is for v0char to be 1, so this last operation doesn't
        # actually do anything unless v0char != 1.
        v0hats = d * (df['v_dot_phihat_disc(T<=1e4)'] / 100.) ** e
        xs = np.concatenate([pdfs[gal]['vs'] / v0hats[gal] * v0char
                             for gal in pdfs])
        ps_truth = np.concatenate([pdfs[gal]['ps'] * v0hats[gal] / v0char
                                   for gal in pdfs])
        model = lmfit.model.Model(calc_p, 
                                  independent_vars=['vs'])
        params = model.make_params()
        params['d'].set(value=d, vary=True, min=0.)
        params['h'].set(value=1., vary=True, min=0.)
        params['k'].set(value = 0.03 * 230. / v0char, vary=True, min=0.)

        print('Fitting.')
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    'ignore', 
                    category=scipy.integrate.IntegrationWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            result = model.fit(ps_truth, params, vs=xs)
        display(result)
        return result

    fig = plt.figure()
    ax = fig.add_subplot(111)
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        universal_fit = pickle.load(f)
    d = universal_fit['d']
    e = universal_fit['e']
    j = universal_fit['j']
    for gal in pdfs:
        v0hat = d * (df.loc[gal, 'v_dot_phihat_disc(T<=1e4)'] / 100.) ** e
        #v0hat = df.loc[gal, 'v0']
        dict_gal = pdfs[gal]
        ax.stairs(dict_gal['ps'] * v0hat / v0char, 
                  dict_gal['bins'] / v0hat * v0char, 
                  zorder=2000)
    # The circular velocity that would put the peak at v0char, given d and e:
    vc_char = 100. * (v0char / d) ** (1. / e) 
    if v0char == 1.:
        std_result = lmfit.minimizer.MinimizerResult()
        std_result.params = dict(d=114.959761,
                                 h=7.60952678,
                                 k=6.66508641)
        covar = np.array([[ 0.89190783, -0.04227626],
                          [-0.04227626,  0.00693924]])
    else:
        std_result = get_std_vdamp_k(vc_char)
        covar = std_result.covar[:-1,:-1]
    # d_std can be either the original d or the best fit d' (d' ~ d  
    # if there is damping. d' = d if there
    # is no damping.) 
    # depending on whether get_std_vdamp() varies the d parameter.
    d_std = std_result.params['d']
    # h_st and k_std are best fits.
    h_std = std_result.params['h']
    k_std = std_result.params['k']
    x0 = d_std * (vc_char/100.) ** e #x0 ~ v0char
    xdamp = h_std * (vc_char/100.) ** j
    xs = np.linspace(0., v0char*2.8, 100)

    ax.plot(xs, smooth_step_max(xs, x0, xdamp, k_std), 
            lw=3., color='k', zorder=3e3)

    ps_samples = make_samples(N_samples, xs, vc_char, d_std, e, h_std, j, 
                              k_std, covar,
                              ddfrac, dhfrac, assume_corr=assume_corr)

    P_1std = scipy.special.erf(1. / np.sqrt(2.)) 
    lower_q = (1. - P_1std) / 2. 
    upper_q = lower_q + P_1std
    lower_q *= 100.
    upper_q *= 100.
    lowers = np.percentile(ps_samples, lower_q, axis=0)
    uppers = np.percentile(ps_samples, upper_q, axis=0)

    ax.fill_between(xs, lowers, uppers, color='grey', alpha=0.6, ec=None)

    ax.axvline(v0char, color='r', ls='--')
    ax.axvline(xdamp, color='r', ls='--')
    plt.show()
    ###########################################################################

    return None

def make_samples(N, vs, vc, d, e, h, j, k, covar, ddfrac, dhfrac, 
                 assume_corr=False, dvc=0.):
    D_DEVS = np.random.normal(0., d * ddfrac, size=N)
    H_DEVS = np.random.normal(0., h * dhfrac, size=N)
    DEVS_UNCORR = np.array([D_DEVS, H_DEVS])
    MU = np.array([[d], [h]])
    if assume_corr:
        c = scipy.linalg.cholesky(covar, lower=True)
    else:
        c = np.identity(2)
    THETA = np.dot(c, DEVS_UNCORR) + MU

    if dvc != 0.:
        # Turning `vc` into a random samples of circular velocities with
        # a std deviation of dvc.
        vc = np.random.normal(vc, dvc, size=N)

    D = THETA[0]
    V0HAT = D * (vc/100.) ** e

    H = THETA[1]
    VDAMP = H * (vc/100.) ** j

    ps_samples = np.array([smooth_step_max(vs,
                                           v0, vdamp, k)
                           for v0, vdamp in zip(V0HAT, VDAMP)])
    return ps_samples

def make_samples_mao(N, vs, vc, vesc, d, e, p, ddfrac, dpfrac, dvc=0.):
    D_DEVS = np.random.normal(0., d * ddfrac, size=N)
    P_DEVS = np.random.normal(0., p * dpfrac, size=N)
    DEVS_UNCORR = np.array([D_DEVS, P_DEVS])
    MU = np.array([[d], [p]])
    c = np.identity(MU.shape[0])
    THETA = np.dot(c, DEVS_UNCORR) + MU 

    if dvc != 0.:
        # Turning `vc` into a random samples of circular velocities with
        # a std deviation of dvc.
        vc = np.random.normal(vc, dvc, size=N)

    D = THETA[0]
    V0HAT = D * (vc / 100.) ** e

    P = THETA[1]

    ps_samples = mao(vs, V0HAT, vesc, P)

    return ps_samples

def save_samples(df_source, N=5000):
    import dm_den
    df = dm_den.load_data(df_source)
    gals = list(df.index)
    for gal_ in ['m12z', 'm12w']:
        gals.remove(gal_)
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        fit = pickle.load(f)
    samples_dict = {}
    vs = np.linspace(0., 700., 300)
    samples_dict['vs'] = vs
    grid_results = grid_eval.identify()
    pbar = ProgressBar()
    with h5py.File(paths.data + 'samples_dz1.0_sigmoid_damped.h5', 'w') as f:
        f.create_dataset('vs', data=vs)
        for gal in pbar(gals):
            vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
            samples_dict[gal] = make_samples(N, vs, vc, 
                                             d=fit['d'], e=fit['e'], 
                                             h=fit['h'],
                                             j=fit['j'], k=fit['k'], 
                                             covar=fit['covar'],
                                             ddfrac=grid_results[0],
                                             dhfrac=grid_results[1], 
                                             assume_corr=False, dvc=0.)
            f.create_dataset(gal, data=samples_dict[gal])
    return samples_dict

def save_samples_mao(df_source, N=5000):
    import dm_den
    df = dm_den.load_data(df_source)
    gals = list(df.index)
    for gal_ in ['m12z', 'm12w']:
        gals.remove(gal_)
    with open(paths.data + 'results_mao_lim_fit.pkl', 'rb') as f:
        params = pickle.load(f)
    vcut_dict = dm_den.load_vcuts('lim_fit', df)
    samples_dict = {}
    vs = np.linspace(0., 700., 300)
    samples_dict['vs'] = vs
    ddfrac, dpfrac = grid_eval_mao.identify('grid_mao.h5')
    pbar = ProgressBar()
    with h5py.File(paths.data + 'samples_dz1.0_mao.h5', 'w') as f:
        f.create_dataset('vs', data=vs)
        for gal in pbar(gals):
            vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
            vcut = vcut_dict[gal]
            samples_dict[gal] = make_samples_mao(N, vs, vc, vcut, params['d'],
                                                 params['e'], params['p'], 
                                                 ddfrac=ddfrac,
                                                 dpfrac=dpfrac,
                                                 dvc=0.)
            f.create_dataset(gal, data=samples_dict[gal])
    return samples_dict

def load_samples(fname='samples_dz1.0_sigmoid_damped.h5'):
    samples_dict = {}
    with h5py.File(paths.data + fname, 'r') as f:
        for key in f.keys():
            if type(f[key]) == h5py.Dataset:
                samples_dict[key] = f[key][()]
            elif type(f[key]) == h5py.Group:
                grp = f[key]
                samples_dict[key] = {}
                for subkey in grp:
                    samples_dict[key][subkey] = grp[subkey][()]

    return samples_dict

def gal_bands(gal, vs, df, result, ddfrac=0.1, dhfrac=0.18,
              assume_corr=False, ax=None, samples_color=plt.cm.viridis(0.5),
              dvc=0.):
    '''
    Generate the upper and lower confidence band given fractional uncertainty
    in d and h. The method does not draw the band but returns the upper and
    lower limits of the band. However, if a matplotlib.axes is provided, it
    will draw the individual distribution samples used to generate the band.

    Parameters
    ----------
    gal: str
        Which galaxy to analyze
    vs: np.ndarray
        Velocities for which to generate the bands        
    df: pd.DataFrame
        Analysis results
    result: lmfit.Minimizer.MinimizerResult, lmfit.model.ModelResult, or dict
        Universal regression result
    ddfrac: float
        Fractional uncertainty on d parameter
    dhfrac: float
        Fractional uncertainty on h parameter
    assume_corr: bool
        Specifies whether to assume d and h are correlated
    ax: matplotlib.axes
        Subplot object into which individual sample distributions are drawn.
        If not provided, individual samples are not draw
    samples_color: str or tuple, default matplotlib.pyplot.cm.viridis(0.5)
        The color to use for the distribution samples
    dvc: float, default 0.
        The uncertainty to assume in the circular speed, in km/s.

    Returns
    -------
    lowers: np.ndarray
        Lower limits of the confidence band at each v in vs
    uppers: np.ndarray
        Upper limits of the confidence band at each v in vs
    '''

    if isinstance(result, (lmfit.minimizer.MinimizerResult, 
                           lmfit.model.ModelResult)):
        d = result.params['d'].value
        e = result.params['e'].value
        h = result.params['h'].value
        j = result.params['j'].value
        k = result.params['k'].value
        covar = result.covar
    elif isinstance(result, dict):
        d = result['d']
        e = result['e']
        h = result['h']
        j = result['j']
        k = result['k']
        covar = result['covar']
    else:
        raise ValueError('Unexpected result type')

    vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
    v0hat = d * (vc/100.) ** e
    vdamp = h * (vc/100.) ** j

    N_samples = 5000

    ps_samples = make_samples(N_samples, vs, vc, 
                              d, e, h, j, k, covar, ddfrac, dhfrac, 
                              assume_corr=assume_corr, dvc=dvc)

    lowers, uppers = gal_bands_from_samples(vs, ps_samples,  
                                            samples_color, ax)

    return lowers, uppers 
     
def gal_bands_from_samples(vs, ps_samples, samples_color, ax=None):
    N_samples = len(ps_samples)
    P_1std = scipy.special.erf(1. / np.sqrt(2)) # ~68%
    lower_q = (1. - P_1std) / 2. 
    upper_q = lower_q + P_1std
    lower_q *= 100.
    upper_q *= 100.
    lowers = np.percentile(ps_samples, lower_q, axis=0)
    uppers = np.percentile(ps_samples, upper_q, axis=0)

    if ax is not None:
        ax.plot(vs, ps_samples.T, color=samples_color, alpha=1.e2/N_samples, 
                zorder=0)

    return lowers, uppers

def three_d_distribs():
    import dm_den
    df = dm_den.load_data('dm_stats_dz1.0_20230626.h5').drop(['m12w', 'm12z'])
    #if gals != 'discs' and not isinstance(gals, (list, np.ndarray)):
    #    raise ValueError('Unexpected value provided for gals arg')
    pdfs = copy.deepcopy(pdfs_v)
    pdfs.pop('m12z')
    pdfs.pop('m12w')

    for gal in pdfs:
        dict_gal = pdfs[gal]
        bins = dict_gal['bins']
        dict_gal['vcirc'] = np.repeat(
            df.loc[gal, 'v_dot_phihat_disc(T<=1e4)'],
            len(dict_gal['ps']))
        dict_gal['vesc'] = np.repeat(vesc_dict[gal]['ve_avg'],
                                     len(dict_gal['ps']))

    indices = np.argsort(df['v_dot_phihat_disc(T<=1e4)'].values)
    ps_truth = np.array([pdfs[galname]['ps']
                         for galname in pdfs])[indices,:]
    ps_truth_smooth = [scipy.signal.savgol_filter(pdfs[gal]['ps'],
                                                  20, 3) 
                       for gal in pdfs]
    ps_truth_smooth = np.array(ps_truth_smooth)[indices,:]
    vs_truth = np.array([pdfs[galname]['vs']
                         for galname in pdfs])[indices,:]
    vcircs = np.array([pdfs[galname]['vcirc']
                       for galname in pdfs])[indices,:]
    
    vs_postfit = np.linspace(0., 700., 1000)

    ###########################################################################
    # Smoothing the data plot
    ###########################################################################
    vcircs_set = df['v_dot_phihat_disc(T<=1e4)'].values
    vcircs_postfit = np.linspace(vcircs_set.min(), vcircs_set.max(), 1000)
    X, Y = np.meshgrid(vs_postfit, vcircs_postfit)

    #m = 600.
    #s = np.mean([m-np.sqrt(2.*m), 
    #             m+np.sqrt(2.*m)])
    
    data = np.array([vs_truth, vcircs, ps_truth])
    filtered = scipy.ndimage.gaussian_filter(data, sigma=[0., 0.8, 0.])
    
    ps_truth_smooth = scipy.interpolate.griddata((vs_truth.flatten(), 
                                                  vcircs.flatten()), 
                                                 ps_truth.flatten(),
                                                 (vs_postfit[None,:], 
                                                  vcircs_postfit[:,None]),
                                                 method='linear')
    interped = np.array([X, Y, ps_truth_smooth])
    filtered_interp = scipy.ndimage.gaussian_filter(interped, 
                                                    sigma=[0., 25., 0.7])

    # Plot the interpolated data with a gaussian filter applied to the
    # interpolation
    fig = plt.figure(dpi=190, figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(*filtered_interp, 
                    cmap=mpl.cm.coolwarm, rcount=100,
                    ccount=300, antialiased=False)
    ax.view_init(elev=33., azim=-102.)
    ax.set_ylabel('\n$v_\mathrm{c}\ \mathrm{\left[km\,s^{-1}\\right]}$')
    ax.set_xlabel('\n$v\ \mathrm{\left[km\,s^{-1}\\right]}$', y=-100)
    ax.set_zlabel('\n$f(v)\,4\pi v^2\ \mathrm{\left[km^{-1}\,s\\right]}$',
                  linespacing=3.)
    zlim = ax.get_zlim()
    plt.show()
    ###########################################################################

    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        params = pickle.load(f)

    def plot_resids():
        v0s = np.array([params['d'] * (vc/100.) ** params['e'] 
                        for vc in vcircs.flatten()])
        vdamps = np.array([params['h'] * (vc/100.) ** params['j']
                           for vc in vcircs.flatten()])
        ps_predicted = np.array([smooth_step_max(v, v0, vdamp, params['k'])
                                 for v, v0, vdamp in zip(vs_truth.flatten(),
                                                         v0s, vdamps)])
        resids = ps_predicted - ps_truth.flatten()

        resids_smooth = scipy.interpolate.griddata((vs_truth.flatten(),
                                                    vcircs.flatten()),
                                                   resids.flatten(),
                                                   (vs_postfit[None,:],
                                                    vcircs_postfit[:,None]),
                                                   method='linear')

        X, Y = np.meshgrid(vs_postfit, vcircs_postfit)

        fig = plt.figure(dpi=190, figsize=(7,6))
        ax = fig.add_subplot(111, projection = '3d')
        ax.plot_surface(X, Y, resids_smooth, cmap=mpl.cm.coolwarm)
        ax.set_zlim(-zlim[1], zlim[1]) 
        ax.set_ylabel('\n$v_\mathrm{c}\ \mathrm{\left[km\,s^{-1}\\right]}$')
        ax.set_xlabel('\n$v\ \mathrm{\left[km\,s^{-1}\\right]}$', y=-100)
        ax.set_zlabel('\nresids $\mathrm{\left[km^{-1}\,s\\right]}$',
                      linespacing=3.)
        ax.view_init(elev=7., azim=-86.)
        plt.show()

        return None

    plot_resids()

    # Generate the prediction
    vcircs = df['v_dot_phihat_disc(T<=1e4)'].values
    vcircs = np.sort(vcircs)
    X, Y = np.meshgrid(vs_postfit, vcircs)
    V0 = params['d'] * Y/100. ** params['e']
    Vdamp = params['h'] * Y/100. **params['j']
    Z = [[smooth_step_max(v, 
                          params['d'] * (vc/100.) ** params['e'],
                          params['h'] * (vc/100.) ** params['j'],
                          params['k']) 
          for v in vs_postfit]
         for vc in vcircs]
    Z = np.array(Z)

    # Plot a wireframe of the prediction
    fig, ax = plt.subplots(dpi=190, figsize=(6,5),
                           subplot_kw={"projection": "3d"})
    surf = ax.plot_wireframe(X, Y, Z)
    plt.show()

    ###########################################################################
    # Plot the colormapped prediction surface
    ###########################################################################
    # Generate the prediction
    vs_postfit = np.linspace(0., 650., 100)
    vcircs_set = df['v_dot_phihat_disc(T<=1e4)'].values
    vcircs_postfit = np.linspace(vcircs_set.min(), vcircs_set.max(), 100)
    #X, Y = np.meshgrid(vs_postfit, vcircs_postfit)

    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        params = pickle.load(f)
    V0 = params['d'] * Y/100. ** params['e']
    Vdamp = params['h'] * Y/100. **params['j']

    vs_postfit, vcircs_postfit = np.meshgrid(vs_postfit, vcircs_postfit)
    vs_postfit = vs_postfit.flatten()
    vcircs_postfit = vcircs_postfit.flatten()
    v0s = params['d'] * (vcircs_postfit/100.) ** params['e']
    vdamps = params['h'] * (vcircs_postfit/100.) ** params['j']
    zs = [smooth_step_max(v, v0, vdamp, params['k'])
          for v, v0, vdamp in zip(vs_postfit, v0s, vdamps)]
    zs = np.array(zs)

    vs_smooth = np.linspace(0., 650., 1000)
    vcircs_smooth = np.linspace(vcircs_set.min(), vcircs_set.max(), 1000)

    Z_smooth = scipy.interpolate.griddata(
            (vs_postfit.flatten(), 
             vcircs_postfit.flatten()), 
            zs.flatten(),
            (vs_smooth[None,:], 
             vcircs_smooth[:,None]),
            method='linear')
    X, Y = np.meshgrid(vs_smooth, vcircs_smooth)

    # Plot the prediction
    fig = plt.figure(dpi=190, figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z_smooth,
                    cmap=mpl.cm.coolwarm, rcount=100,
                    ccount=300, antialiased=False)
    ax.view_init(elev=33., azim=-102.)
    ax.set_ylabel('\n$v_\mathrm{c}\ \mathrm{\left[km\,s^{-1}\\right]}$')
    ax.set_xlabel('\n$v\ \mathrm{\left[km\,s^{-1}\\right]}$', y=-100)
    ax.set_zlabel('\n$f(v)\,4\pi v^2\ \mathrm{\left[km^{-1}\,s\\right]}$',
                  linespacing=3.)
    ax.view_init(elev=23., azim=-97.)
    plt.show()

def diff_fr68(params, assume_corr=False, incl_area=True, 
              verbose=False):
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        data_raw = pickle.load(f)
    pdfs = copy.deepcopy(pdfs_v)
    del pdfs['m12w']
    del pdfs['m12z']

    def count_within(gal, ddfrac, dhfrac):
        import dm_den
        df = dm_den.load_data('dm_stats_dz1.0_20230626.h5')

        pdf_gal = pdfs[gal]
        ps = pdf_gal['ps']
        vs = pdf_gal['vs']
        lowers, uppers = gal_bands(gal, vs, df, data_raw, ddfrac, 
                                   dhfrac, assume_corr=assume_corr)
        is_above = ps > uppers
        is_below = ps < lowers
        is_outlier = is_above | is_below 
        N_out = np.sum(is_outlier)
        N_tot = len(vs)
        
        percent = 1. - N_out / N_tot

        area = scipy.integrate.simpson(uppers, vs)
        area -= scipy.integrate.simpson(lowers, vs)

        return percent, area

    ddfrac = params['ddfrac']
    dhfrac = params['dhfrac']
    percents_areas = np.array([count_within(gal, ddfrac, dhfrac) \
                                for gal in pdfs])
    percents_within, areas = percents_areas.T
    P_1std = scipy.special.erf(1. / np.sqrt(2.)) # ~68%
    resids_vals = percents_within - P_1std 
    if incl_area:
        resids_vals = np.concatenate((resids_vals, areas))
    if verbose:
        print('ddfrac = {0:0.4f}, '
              'dhfrac = {1:0.4f}, '
              'SSE = {2:0.3f}, '
              'frac within = {3:0.4f}, '
              'fracs within {4}'.format(ddfrac.value, dhfrac.value, 
                                                 np.sum(resids_vals ** 2.), 
                                                 percents_within.mean(),
                                                 percents_within))
    return resids_vals

def count_within_agg(ddfrac, dhfrac, df, assume_corr=False, 
                     return_fracs=False, data_override=None):
    if data_override is not None:
        data_raw = data_override
    else:
        with open(paths.data + 'data_raw.pkl', 'rb') as f:
            data_raw = pickle.load(f)
    pdfs = copy.deepcopy(pdfs_v)
    del pdfs['m12w']
    del pdfs['m12z']

    N_out = 0.
    N_tot = 0.
    area = 0.
    area_norm = 0.

    for gal in pdfs:
        pdf_gal = pdfs[gal]
        ps = pdf_gal['ps']
        vs = pdf_gal['vs']

        lowers, uppers = gal_bands(gal, vs, df, data_raw, ddfrac, 
                                   dhfrac, assume_corr=assume_corr)

        area += scipy.integrate.simpson(uppers, vs)
        area -= scipy.integrate.simpson(lowers, vs)
        # Area under the truth distribution:
        # (although I just realized this will equal 1 for each galaxy)
        area_norm += scipy.integrate.simpson(ps, vs)

        is_above = ps > uppers
        is_below = ps < lowers
        is_outlier = is_above | is_below 
        N_out += np.sum(is_outlier)
        N_tot += len(vs)
    
    percent = 1. - N_out / N_tot
    # Normalize the area cost by the total area under all the truth
    # distributions so the area cost doesn't dominate the diff-from-68
    # cost.
    area /= area_norm
    # Also need to further reduce the weight of area because we're too far
    # from 68% otherwise.
    area /= 10. 
    if return_fracs:
        return percent, area, ddfrac, dhfrac
    else:
        return percent, area

def count_within_agg_mao(ddfrac, dpfrac, df, params):
    import dm_den
    pdfs = copy.deepcopy(pdfs_v)
    del pdfs['m12w']
    del pdfs['m12z']

    N_out = 0.
    N_tot = 0.
    area = 0.
    area_norm = 0.
 
    vcuts_dict = dm_den.load_vcuts('lim_fit', df)

    for gal in pdfs:
        pdf_gal = pdfs[gal]
        ps = pdf_gal['ps']
        vs = pdf_gal['vs']
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        vcut = vcuts_dict[gal]

        ps_samples = make_samples_mao(5000, vs, vc, vcut, params['d'],
                                      params['e'], params['p'], ddfrac, 
                                      dpfrac, dvc=0.)
        lowers, uppers = gal_bands_from_samples(vs, ps_samples, 
                                                samples_color=None, ax=None)

        area += scipy.integrate.simpson(uppers, vs)
        area -= scipy.integrate.simpson(lowers, vs)
        # Area under the truth distribution:
        # (although I just realized this will equal 1 for each galaxy)
        area_norm += scipy.integrate.simpson(ps, vs)

        is_above = ps > uppers
        is_below = ps < lowers
        is_outlier = is_above | is_below 
        N_out += np.sum(is_outlier)
        N_tot += len(vs)
    
    percent = 1. - N_out / N_tot
    # Normalize the area cost by the total area under all the truth
    # distributions so the area cost doesn't dominate the diff-from-68
    # cost.
    area /= area_norm
    # Also need to further reduce the weight of area because we're too far
    # from 68% otherwise.
    area /= 10. 
    return percent, area, ddfrac, dpfrac

def diff_fr68_agg(params, assume_corr=False, incl_area=True,
                  verbose=False):
    df = pd.read_pickle(paths.data + 'dm_stats_dz1.0_20230626.pkl')

    ddfrac = params['ddfrac']
    dhfrac = params['dhfrac']
    percent_within, area = count_within_agg(ddfrac, dhfrac, df, assume_corr)
    P_1std = scipy.special.erf(1. / np.sqrt(2.)) # ~68%
    diff = percent_within - P_1std 
    if incl_area:
        cost = np.array([diff, area])
        cost2 = cost**2.
    else:
        # I'm just squaring the cost and giving cost and cost2 the same value
        # here because I don't feel like puting the
        # return in a conditional.
        cost = diff**2. 
        cost2 = cost
    if verbose:
        print('ddfrac = {0:0.4f}, '
              'dhfrac = {1:0.4f}, '
              'SSE = {2:0.3e}, '
              'frac within = {3:0.4f}, '
              'area = {4:0.4f}'.format(ddfrac.value, dhfrac.value, 
                                       np.sum(cost2), 
                                       percent_within, area))
    return cost

def find_68_uncertainty(method, assume_corr=False, diff_fcn=diff_fr68,
                        incl_area=True, verbose=False, **kwargs):
    start = time.time()

    pdfs = copy.deepcopy(pdfs_v)
    del pdfs['m12w']
    del pdfs['m12z']

    params = lmfit.Parameters()
    params.add('ddfrac', value=0.15, vary=True, min=0., max=0.5)
    params.add('dhfrac', value=0.15, vary=True, min=0., max=0.5)

    minimizer_result = lmfit.minimize(diff_fcn, params, method=method,
                                      kws=dict(assume_corr=assume_corr,
                                               incl_area=incl_area,
                                               verbose=verbose),
                                      **kwargs)

    elapsed = time.time() - start
    minutes = elapsed // 60.
    sec = elapsed - minutes * 60.
    print('{0:0.0f}min, {1:0.1f}s taken to optimize.'.format(minutes, sec)) 

    return minimizer_result

###############################################################################
def vs_into_bins(vs):
    diffs = np.diff(vs)
    diff = diffs[0]
    assert np.all(np.isclose(diffs, diff))
    bins = vs - diff / 2.
    bins = np.append(bins, bins[-1] + diff)
    vs_from_bins = (bins[1:] + bins[:-1]) / 2.
    assert np.allclose(vs, vs_from_bins)
    return bins

def determine_systematics(
        df_source,
        distrib_samples_fname=('mcmc_distrib_samples'
                               '_by_v0_20240702'
                               '(narrower_uniform_prior_20240606)'
                               '.h5'), 
        v_by_v0_pdf_fname='v_by_v0_pdfs_disc_dz1.0.pkl',
        verbose=False,
        update_paper=False):
    import dm_den
    import dm_den_viz
    import fitting
    '''
    Given a sample of parameters, which carry an implied statistical error,
    determine the additional systematic error to explain the remainder of the
    deviation of the prediction from the data.

    Parameters
    ----------
    df_source: str
        File name of the analysis results DataFrame.
    distrib_samples_fname: str, default ('mcmc_distrib_samples'
                                         '_by_v0_narrower_uniform_prior'
                                         '_20240606.h5')
        File name of the speed distribution samples from `emcee`.
    v_by_v0_pdf_fname: str, default 'v_by_v0_pdfs_disc_dz1.0.pkl'
        Name of the file 
        containing probability densities of, as usual, speed, but with bins 
        based on each
        galaxy's v0. Each galaxy will have different bins in v but the same 
        bins in
        v/v0. The purpose of this is to be able to calculate systematic errors
        of the final model as a function of v/v0.
    verbose: bool, default False
        Whether to print information about how dominant systematics are.
    update_paper: bool, default False
        Whether to update the csv that the paper reads.
    
    Returns
    -------
    vs_by_v0: np.ndarray
        The one v/v0 array used by all galaxies in v_by_v0_fname.
    tot_errs: np.ndarray
        The total uncertainty corresponding to each v/v0.
    sys_errs: np.ndarray
        The systematic portion of the uncertainty at each v/v0.
    '''
    
    df = dm_den.load_data(df_source).drop(['m12z', 'm12w'])
    samples = load_samples(distrib_samples_fname)
    d, e, h, j, k = (samples['params'][t] 
                     for t in ['d', 'e', 'h', 'j', 'k'])
    vs_by_v0_distrib_samples = samples['v_v0']

    Y = [] # Probability densities from the data
    YHAT = [] # Predicted probablity densities from the fit
    LOWER_STAT = [] # Bottom of statistical error band
    UPPER_STAT = [] # Top of statistical error band  
    v0s = []

    with open(paths.data + v_by_v0_pdf_fname, 'rb') as f:
        pdfs = pickle.load(f)

    for i, galname in enumerate(df.index):
        vc = df.loc[galname, 'v_dot_phihat_disc(T<=1e4)']
        v0 = d * (vc / 100.) ** e 
        vdamp = h * (vc / 100.) ** j
        v0s.append(v0)
        vs = samples[galname]['vs']

        pdf = pdfs[galname]
        vs_by_v0_pdf = pdf['vs_by_v0']

        if not np.array_equal(vs_by_v0_distrib_samples, vs_by_v0_pdf):
            print(np.array([vs_by_v0_pdf, vs_by_v0_distrib_samples]).T)
            raise ValueError(
                'v/v0 array from the PDFs file does not match v/v0 from the'
                ' distribution samples file.')

        ps = pdf['ps']
        Y.append(ps)

        lowers, uppers = gal_bands_from_samples(
            vs_by_v0_distrib_samples, 
            samples[galname]['ps'], 
            plt.cm.viridis(0.5),
        )
        UPPER_STAT.append(uppers)
        LOWER_STAT.append(lowers)

        yhat = fitting.smooth_step_max(
                vs,
                v0,
                vdamp,
                k
        )
        YHAT.append(yhat)
    Y = np.array(Y)
    YHAT = np.array(YHAT)
    LOWER_STAT = np.array(LOWER_STAT)
    UPPER_STAT = np.array(UPPER_STAT)
    v0s = np.array(v0s)

    ###########################################################################
    # Total error band, of which systematic errors make >97%
    ###########################################################################
    TOT_ERR = YHAT - Y
    tot_errs = np.sqrt((TOT_ERR.T ** 2.).mean(axis=1)) # RMS

    ###########################################################################
    # Isolate the systematic portion of the uncertainty band
    ###########################################################################
    # Distance of the lower and upper statistical bounds from
    # the target vector:
    dist_lower = LOWER_STAT - Y
    dist_upper = UPPER_STAT - Y
    abs_dists = np.abs(np.array([dist_lower, dist_upper]))
    dist_indices = np.argmin(
            abs_dists,
            axis = 0,
    )
    # Systematic portion of the deviations from the predictions to the data
    SYS_ERR = np.array([dist_lower, dist_upper])[
        dist_indices,
        np.arange(abs_dists.shape[1])[:, None],
        np.arange(abs_dists.shape[2])[None, :]
    ]
    # If the band captured the data at a certain v/v0 for a certain galaxy,
    # set the systematic error to 0 at that data point.
    is_captured = (LOWER_STAT < Y) & (Y < UPPER_STAT)
    SYS_ERR[is_captured] = 0.
    sys_errs = np.sqrt((SYS_ERR.T ** 2.).mean(axis=1)) # RMS

    # Statistical uncertainties broken out by upper and lower band
    STAT_ERR = (np.array([UPPER_STAT, LOWER_STAT]) - YHAT).transpose(1, 2, 0)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        min_mult = np.min(
            np.repeat(sys_errs[:, np.newaxis], 2, axis=1) / np.abs(STAT_ERR)
        )
    if verbose:
        print('Systematics are at a minimum {0:0.0f}x greater than statistics.'
              .format(math.floor(min_mult)))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            print('Systematics make up at least {0:0.0f}% of the total'
                  ' uncertainty.'
                  .format(math.floor(np.min(sys_errs / tot_errs) * 100.)))
        

    ###########################################################################
    # Things I don't use but might want later
    ###########################################################################
    # Percent difference of systematic portion of deviations from the
    # prediction
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        SYS_PERCENT_ERR = SYS_ERR / YHAT
    SYS_PERCENT_ERR[~np.isfinite(SYS_PERCENT_ERR)] = np.nan 
    sys_percent_errs = np.sqrt((SYS_PERCENT_ERR.T ** 2.).mean(axis=1)) # RMS

    # Statistical uncertainties averaging upper and lower
    STAT_ERR_AVG = (UPPER_STAT - LOWER_STAT) / 2. 

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        STAT_PERCENT_ERR = STAT_ERR / np.repeat(YHAT[:, :, np.newaxis], 
                                                2, 
                                                axis=2)
    ###########################################################################

    if update_paper:
        np.savetxt(
            paths.tables + 'errors.csv',
            np.array([vs_by_v0_distrib_samples, tot_errs / 1.e-3]).T,
            fmt='%0.2f,%0.2f'
        )

    return (
        vs_by_v0_distrib_samples,
        tot_errs,
        sys_errs, 
    )
