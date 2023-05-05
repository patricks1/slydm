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
import numpy as np
import pandas as pd
import multiprocessing as mp
from progressbar import ProgressBar

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['font.family'] = 'serif' 
rcParams['axes.titlesize']=24
rcParams['axes.titlepad']=15
rcParams['legend.frameon'] = True
rcParams['legend.fontsize']=15
rcParams['figure.facecolor'] = (1., 1., 1., 1.) #white with alpha=1.

with open('./data/v_pdfs.pkl','rb') as f:
    pdfs_v=pickle.load(f)
with open('./data/vescs_20221222.pkl', 'rb') as f:
    vesc_dict = pickle.load(f)
with open('./data/v_pdfs_incl_ve_20220205.pkl','rb') as f:
    pdfs_v_incl_vearth=pickle.load(f)

for gal in pdfs_v:
    bins = pdfs_v[gal]['bins']
    vs = (bins[1:] + bins[:-1]) / 2.
    pdfs_v[gal]['vs'] = vs

max_bins_est = 1.e-3
min_bins_est = -1.e-3
resids_lim = 0.7

def pN_smooth_step_max(v, v0, vesc, k):
    '''
    Probability density before normalizing by N
    '''
    fN = np.exp( - v**2. / v0**2. )
    pN = fN * 4. * np.pi * v**2.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 
                                category=scipy.integrate.IntegrationWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        trunc = 1. / (1. + np.exp(-k * (vesc-v)))
    pN *= trunc
    return pN

def smooth_step_max(v, v0, vesc, k):
    '''
    Smooth-step-truncated Maxwellian, as opposed to the immediate cutoff
    of a Heaviside function used in trunc_max
    
    k is the strength of the exponential cutoff
    '''
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        N = scipy.integrate.quad(pN_smooth_step_max, 0., np.inf, 
                                 (v0, vesc, k), epsabs=0)[0]
        p = pN_smooth_step_max(v, v0, vesc, k) / N
    return p

def exp_max(v, v0, vesc):
    '''
    Maxwellian with an exponential decline (from Macabe 2010 and Lacroix et al. 
    2020)
    '''
    fN = fN = np.exp( - v**2. / v0**2. ) - np.exp( - vesc**2. / v0**2. )
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

def label_axes(axs, gals):
    if not isinstance(gals, (list, np.ndarray, pd.core.indexes.base.Index)) \
           and gals == 'discs':
        for i in [4]:
            axs[i].set_ylabel('$f(v)\,4\pi v^2\ [\mathrm{km^{-1}\,s}]$')
        for ax in axs[-4:]:
            ax.set_xlabel('$v\ [\mathrm{km\,s^{-1}}]$')
    elif len(gals) < 4:
        axs[0].set_ylabel('$f(v)\,4\pi v^2\ [\mathrm{km^{-1}\,s}]$')
        for ax in axs:
            ax.set_xlabel('$v\ [\mathrm{km\,s^{-1}}]$')
    return None

def fit_v0(gals='discs', show_exp=False, tgt_fname=None):
    import dm_den
    df = dm_den.load_data('dm_stats_20221208.h5')
    if gals == 'discs':
        df = df.drop(['m12w', 'm12z'])
    elif isinstance(gals, (list, np.ndarray)):
        df = df.loc[gals]
    else:
        raise ValueError('Unexpected value provided for gals arg')
    
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
    
    for i, gal in enumerate(pbar(df.index)):
        pdf = pdfs_v[gal]
        bins = pdf['bins']
        vs_truth = (bins[1:] + bins[:-1]) / 2.
        vs_postfit = np.linspace(0., 750., 500)
        ps_truth = pdf['ps']
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        vesc = vesc_dict[gal]['ve_avg']
    
        p = lmfit.Parameters()
        p.add('v0', value=300., vary=True, min=100., max=400.)
        p.add('vesc', value=np.inf, vary=False)
        p.add('k', value=0.0309, vary=False, min=0.0001, max=1.)
    
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
                res_v0_vesc.params['k']))
        
        if show_exp:
            del(p['k'])
            #p['vesc'].set(value = vesc_dict[gal]['ve_avg'], vary=False)
            res_exp = lmfit.minimize(resids_exp_max, p, method='nelder',
                                     args=(vs_truth, ps_truth), nan_policy='omit')
            axs[i].plot(vs_postfit,
                        exp_max(vs_postfit, 
                                res_exp.params['v0'],
                                res_exp.params['vesc']))
        axs[i].grid(False)
        
        loc=[0.97,0.96]
        kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                      va='top', ha='right', 
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        axs[i].annotate(gal, loc,
                        **kwargs_txt)
        loc[1] -= 0.1
        kwargs_txt['fontsize'] = 11.
        axs[i].annotate(#'$v_\mathrm{{esc}}'
                        #'={0:0.0f}\,\mathrm{{km\,s^{{-1}}}}$\n'
                        '$v_\mathrm{{damp}}'
                        '={1:0.0f}\,\mathrm{{km\,s^{{-1}}}}$\n'
                        '$v_0={3:0.0f}$\n'
                        #'$k={6:0.4f}$\n'
                        '$\chi^2={2:0.2e}$\n'
                        #'N$_\mathrm{{eval}}={5:0.0f}$'
                        .format(vesc, 
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

    label_axes(axs, gals)

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=140)

    plt.show()

    return vesc_fits

def fit_vdamp(gals='discs', show_exp=False, tgt_fname=None):
    '''
    Plot the best posible distributions, individually fitting vdamp and v0 for
    each galaxy

    Parameters
    ----------
    gals: str or list-like of str
        Galaxies to plot
    show_exp: bool
        If True, include the exponentially cutoff form from Macabe 2010 and 
        Lacrois et al. 2020.
    tgt_fname: str
        File name with which to save the plot. Default, None, is to not save
        the plot.

    Return
    ------
    vdamp_fits: dict
        Dictionary keyed by galaxy of the best fit vdamps
    '''
    import dm_den
    df = dm_den.load_data('dm_stats_20221208.h5')
    if gals == 'discs':
        df = df.drop(['m12w', 'm12z'])
    elif isinstance(gals, (list, np.ndarray)):
        df = df.loc[gals]
    else:
        raise ValueError('Unexpected value provided for gals arg')
    Ngals = len(df)    
    if gals == 'discs':
        figsize = (19., 12.)
        Nrows = 3
        Ncols = 4
    else:
        Ncols = min(len(gals), 4)
        Nrows = math.ceil(len(gals) / Ncols)
    xfigsize = 4.6 / 2. * Ncols + 1.
    yfigsize = 1.5 * Nrows + 1. 
    if Ngals == 2:
        # Add room for residual plots
        Nrows += 1
        yfigsize += 1. 
        height_ratios = [4,1]
    else:
        height_ratios = None
    fig,axs=plt.subplots(Nrows, Ncols, figsize=(xfigsize, yfigsize), 
                         sharey='row',
                         sharex=True, dpi=140, height_ratios=height_ratios)
    axs=axs.ravel()
    fig.subplots_adjust(wspace=0.,hspace=0.)
    
    pbar = ProgressBar()
    
    def sse_max_v0_vesc(params, vs_truth, ps_truth):
        v0 = params['v0'].value
        #print(v0)
        vesc = params['vdamp'].value
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
        vesc = params['vdamp'].value
        with np.errstate(divide='ignore'):
            ps_predicted = exp_max(vs_truth,
                                   v0,
                                   vesc)
        resids = ps_predicted - ps_truth
        return resids
        
    vdamp_fits = {}
    
    for i, gal in enumerate(pbar(df.index)):
        pdf = pdfs_v[gal]
        bins = pdf['bins']
        vs_truth = (bins[1:] + bins[:-1]) / 2.
        vs_postfit = np.linspace(0., 750., 500)
        ps_truth = pdf['ps']
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        vesc = vesc_dict[gal]['ve_avg']
    
        p = lmfit.Parameters()
        p.add('v0', value=300., vary=True, min=100., max=400.)
        p.add('vdamp', value=470., vary=True, min=250., max=600.)
        p.add('k', value=0.0309, vary=False, min=0.0001, max=1.)
    
        res_v0_vesc = lmfit.minimize(sse_max_v0_vesc, p, 
                                      method='nelder', 
                                      args=(vs_truth, ps_truth),
                                      nan_policy='omit', 
                                      #niter=300
                                     )
        vdamp_fits[gal] = res_v0_vesc.params['vdamp'].value
        _ = [res_v0_vesc.params[key] 
                                       for key in ['v0', 'vdamp', 'k']]
        rms_err = calc_rms_err(vs_truth, ps_truth, smooth_step_max,
                               args=[res_v0_vesc.params[key] 
                                     for key in ['v0', 'vdamp', 'k']])
        rms_txt = staudt_utils.mprint(rms_err, d=1, 
                                      show=False).replace('$','')
        
        axs[i].stairs(ps_truth, bins, color='k', label='data')
        axs[i].plot(
            vs_postfit, 
            smooth_step_max(
                vs_postfit, 
                res_v0_vesc.params['v0'], 
                res_v0_vesc.params['vdamp'],
                res_v0_vesc.params['k']),
            label='fit', color='C2')
        if show_exp:
            del(p['k'])
            #p['vdamp'].set(value = vesc_dict[gal]['ve_avg'], vary=False)
            res_exp = lmfit.minimize(resids_exp_max, p, method='nelder',
                                     args=(vs_truth, ps_truth), 
                                     nan_policy='omit')
            axs[i].plot(vs_postfit,
                        exp_max(vs_postfit, 
                                res_exp.params['v0'],
                                res_exp.params['vdamp']))
        axs[i].grid(False)
        # Make ticks on both sides of the x-axis:
        axs[i].tick_params(axis='x', direction='inout', length=6)
        order_of_mag = -3
        axs[i].ticklabel_format(style='sci', axis='y', 
                                scilimits=(order_of_mag,
                                           order_of_mag),
                                useMathText=True)

        loc=[0.97,0.96]
        kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                      va='top', ha='right', 
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        axs[i].annotate(gal, loc,
                        **kwargs_txt)
        '''
        loc[1] -= 0.1
        kwargs_txt['fontsize'] = 11.
        axs[i].annotate(#'$v_\mathrm{{esc}}'
                        #'={0:0.0f}\,\mathrm{{km\,s^{{-1}}}}$\n'
                        '$v_\mathrm{{damp}}'
                        '={1:0.0f}\,\mathrm{{km\,s^{{-1}}}}$\n'
                        '$v_0={3:0.0f}$\n'
                        #'$k={6:0.4f}$\n'
                        '$\mathrm{{RMS_{{err}}}}={4:s}$'
                        #'N$_\mathrm{{eval}}={5:0.0f}$'
                        .format(vesc, 
                                res_v0_vesc.params['vdamp'].value,
                                res_v0_vesc.chisqr, 
                                res_v0_vesc.params['v0'].value,
                                rms_txt,
                                res_v0_vesc.nfev,
                                res_v0_vesc.params['k'].value,
                               ),
                        loc, **kwargs_txt)
        if show_exp:
            loc[1] -= 0.2
            axs[i].annotate('$\chi^2_\mathrm{{exp}}={0:0.2e}$\n'
                            .format(res_exp.chisqr),
            loc, **kwargs_txt)
        '''
        if Ngals == 2:
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
            ps_hat = smooth_step_max(vs_truth, 
                                     res_v0_vesc.params['v0'].value,
                                     res_v0_vesc.params['vdamp'].value,
                                     res_v0_vesc.params['k'].value)
            resids = ps_hat - ps_truth
            resids_extend = smooth_step_max(vs_extend, 
                                            res_v0_vesc.params['v0'].value,
                                            res_v0_vesc.params['vdamp'].value,
                                            res_v0_vesc.params['k'].value)
            resids = np.append(resids, resids_extend, axis=0)
            axs[i+2].plot(vs_resids, resids / 10.**order_of_mag, color='C2')
            axs[i+2].axhline(0., linestyle='--', color='k', alpha=0.5, lw=1.)
            axs[i+2].set_ylim(-resids_lim, resids_lim)
            if i == 0:
                axs[i+2].set_ylabel('resids')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.,hspace=0.)
    axs[-4].legend(loc='upper center',
                   bbox_to_anchor=(1., -0.5),
                   #bbox_to_anchor=(0.5, 0.035),
                   #bbox_transform=fig.transFigure, 
                   ncol=2)
    label_axes(axs, gals)
    
    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=250)

    plt.show()

    return vdamp_fits

def calc_rms_err(xs, ys, fcn, args):
    ys_hat = fcn(xs, *args)
    rms = np.sqrt( np.mean( (ys_hat - ys)**2. ) )
    return rms 

def save_rms_errs(rms_dict):
    try:
        with open(paths.data + 'rms_errs.pkl', 'rb') as f:
            dict_last = pickle.load(f)
    except FileNotFoundError:
        dict_last = {}  
    d = dict_last | rms_dict
    with open(paths.data + 'rms_errs.pkl', 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    return None

def plt_naive(gals='discs', tgt_fname=None, update_vals=False, 
              show_sigma_vc=True):
    if update_vals and gals != 'discs':
        raise ValueError('You should only update values when you\'re plotting '
                         'all the discs.')
    import dm_den
    df = dm_den.load_data('dm_stats_20221208.h5')
    if gals == 'discs':
        df.drop(['m12w', 'm12z'], inplace=True)
    elif isinstance(gals, (list, np.ndarray)):
        df = df.loc[gals]
    else:
        raise ValueError('Unexpected value provided for gals arg')

    Ngals = len(df)
    if gals == 'discs':
        figsize = (19., 12.)
        Nrows = 3
        Ncols = 4
    else:
        Ncols = min(Ngals, 4)
        Nrows = math.ceil(len(gals) / Ncols)
    xfigsize = 4.6 / 2. * Ncols + 1.
    yfigsize = 1.5 * Nrows + 1. 
    if Ngals == 2:
        # Add room for residual plots
        Nrows += 1
        yfigsize += 1. 
        height_ratios = [4,1]
    else:
        height_ratios = None
    fig,axs=plt.subplots(Nrows, Ncols, figsize=(xfigsize, yfigsize), 
                         sharey='row',
                         sharex=True, dpi=140, height_ratios=height_ratios)
    axs=axs.ravel()
    
    pbar = ProgressBar()

    # velocities to use when plotting the functional form attempts
    vs_maxwell = np.linspace(0., 800., 700) 
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        # Strength at with to truncate the distribution
        # I'm probably not going to use this here, though
        k = pickle.load(f)['k'] 

    rms_dict = {}
    rms_dict['sigma_vc'] = {}
    rms_dict['true_sigma'] = {}
    for i, gal in enumerate(df.index):
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        with open(paths.data + 'data_raw.pkl', 'rb') as f:
            results_dict = pickle.load(f)
        sigma_predicted = 10.**results_dict['logdisp_intercept'] \
                          * (vc/100.) ** results_dict['disp_slope']
        sigma_truth = df.loc[gal, 'disp_dm_disc_cyl']
        vesc = vesc_dict[gal]['ve_avg']
        bins = pdfs_v[gal]['bins']
        vs_truth = (bins[1:] + bins[:-1]) / 2.
        ps_truth = pdfs_v[gal]['ps']
        # p(sigma(vc), vs_maxwell)
        ps_sigma_vc = smooth_step_max(
                                   vs_maxwell, 
                                   np.sqrt(2./3.)*sigma_predicted,
                                   np.inf,
                                   np.inf)
        rms_err_sigma_vc = calc_rms_err(vs_truth, ps_truth, smooth_step_max,
                                        args=(np.sqrt(2./3)*sigma_predicted,
                                              np.inf, np.inf))
        rms_sigma_vc_txt = staudt_utils.mprint(rms_err_sigma_vc, d=1, 
                                               show=False).replace('$','')
        rms_dict['sigma_vc'][gal] = rms_err_sigma_vc
        # p(sigma_measured, vs_maxwell)
        ps_true_sigma = smooth_step_max(vs_maxwell,
                                        np.sqrt(2./3.) * sigma_truth,
                                        np.inf,
                                        np.inf)
        rms_err_true_sigma = calc_rms_err(vs_truth, ps_truth, smooth_step_max,
                                          args=(np.sqrt(2./3.)*sigma_truth,
                                                np.inf, np.inf))
        rms_true_sigma_txt= staudt_utils.mprint(rms_err_true_sigma, d=1, 
                                            show=False).replace('$','')
        rms_dict['true_sigma'][gal] = rms_err_true_sigma
        axs[i].stairs(ps_truth, bins,
                      color='k', label='data')
        if show_sigma_vc:
            axs[i].plot(vs_maxwell, ps_sigma_vc, 
                        label='$\sigma_\mathrm{3D}(v_\mathrm{c})$'
                       )
        axs[i].plot(vs_maxwell, ps_true_sigma,
                    label = '$v_0=\sqrt{2/3}\sigma_\mathrm{3D,meas}$')

        # Draw vesc line
        axs[i].axvline(vesc, ls='--', alpha=0.5, color='grey')
        trans = mpl.transforms.blended_transform_factory(axs[i].transData,
                                                         axs[i].transAxes)
        axs[i].text(vesc, 0.5, '$v_\mathrm{esc}(\Phi)$', transform=trans,
                    fontsize=15., rotation=90., color='gray', 
                    horizontalalignment='right')

        axs[i].grid(False)
        # Make ticks on both sides of the x-axis:
        axs[i].tick_params(axis='x', direction='inout', length=6)
        order_of_mag = -3
        axs[i].ticklabel_format(style='sci', axis='y', 
                                scilimits=(order_of_mag,
                                           order_of_mag),
                                useMathText=True)

        loc=[0.97,0.96]
        kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                      va='top', ha='right', 
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        axs[i].annotate(gal, loc,
                        **kwargs_txt)
        '''
        loc[1] -= 0.12
        kwargs_txt['fontsize'] = 10.
        axs[i].annotate('$\mathrm{{RMS}}_{{\sigma'
                            '(v_\mathrm{{c}})}}={0:s}$'
                        .format(rms_sigma_vc_txt),
                        loc, color='C0', **kwargs_txt)
        loc[1] -= 0.11
        axs[i].annotate('$\mathrm{{RMS}}_{{'
                            '\sigma_\mathrm{{meas}}'
                            '}}={0:s}$'
                        .format(rms_true_sigma_txt),
                        loc, color='C1', **kwargs_txt)
        '''
        if Ngals == 2:
            # Draw residual plot
            vs_resids = copy.deepcopy(vs_truth)
            vs_extend = np.linspace(vs_resids.max(), vs_maxwell.max(), 20)
            vs_resids = np.append(vs_resids, vs_extend, axis=0)                                         

            def calc_resids(sigma):
                ps_sigma = smooth_step_max(
                        vs_truth,
                        np.sqrt(2./3.) * sigma,
                        np.inf,
                        np.inf)
                resids = ps_sigma - ps_truth
                inrange = (vs_truth > 75.) & (vs_truth < 175.)
                resids_extend = smooth_step_max(
                    vs_extend,
                    np.sqrt(2./3.) * sigma,
                    np.inf, np.inf)
                resids = np.append(resids, 
                                   resids_extend,
                                   axis=0)
                return resids
            # Remove the 0 tick label because of overlap
            y0, y1 = axs[i].get_ylim()
            visible_ticks = np.array([t for t in axs[i].get_yticks() \
                                      if t>=y0 and t<=y1])
            new_ticks = visible_ticks[visible_ticks > 0.]
            axs[i].set_yticks(new_ticks)

            axs[i+2].grid(False)
            axs[i+2].set_ylim(-resids_lim, resids_lim)

            if show_sigma_vc:
                axs[i+2].plot(vs_resids, 
                              calc_resids(sigma_predicted)/10.**order_of_mag)
            axs[i+2].axhline(0., linestyle='--', color='k', alpha=0.5,
                             lw=1.)
            axs[i+2].plot(vs_resids, 
                          calc_resids(sigma_truth)/10.**order_of_mag)
            axs[i+2].axvline(vesc, ls='--', alpha=0.5, color='grey')

            if i == 0:
                axs[i+2].set_ylabel('resids')
    if update_vals:
        save_rms_errs(rms_dict)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.,hspace=0.)
    if show_sigma_vc:
        axs[1].legend(loc='upper right', bbox_to_anchor=(1., -0.04),
                      bbox_transform=fig.transFigure, ncol=3)
    else:
        trans_legend = mpl.transforms.blended_transform_factory(
                axs[1].transAxes, fig.transFigure)
        axs[1].legend(loc='upper center', bbox_to_anchor=(0., -0.04),
                      bbox_transform=trans_legend, ncol=2)
    label_axes(axs, gals)

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=250)

    plt.show()

    return None

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
    df = dm_den.load_data('dm_stats_20221208.h5').drop(['m12w', 'm12z'])
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

        label_axes(axs, gals)

        if tgt_fname is not None:
            plt.savefig(paths.figures+tgt_fname,
                        bbox_inches='tight',
                        dpi=140)

        plt.show()

    return result_dehjk

def setup_universal_fig(gals):
    if gals == 'discs':
        figsize = (19., 12.)
        Nrows = 3
        Ncols = 4
        Ngals = 12
    else:
        Ncols = min(len(gals), 4)
        Nrows = math.ceil(len(gals) / Ncols)
        Ngals = len(gals)
    xfigsize = 4.6 / 2. * Ncols + 1.
    yfigsize = 1.5 * Nrows + 1. 
    if Ngals == 2:
        # Add room for residual plots
        Nrows += 1
        yfigsize += 1. 
        height_ratios = [4,1]
    else:
        height_ratios = None
    fig,axs=plt.subplots(Nrows, Ncols, figsize=(xfigsize, yfigsize), 
                         sharey='row',
                         sharex=True, dpi=140, height_ratios=height_ratios)
    axs=axs.ravel()
    fig.subplots_adjust(wspace=0.,hspace=0.)
    fig.Nrows = Nrows

    return fig, axs

def plt_mw(tgt_fname=None):
    import dm_den_viz
    import grid_eval
    import dm_den
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        results = pickle.load(f)
    ddfrac, dhfrac = grid_eval.identify()

    vc = dm_den_viz.vc_eilers
    vs = np.linspace(0., 750., 300)

    def predict(vc, ax, **kwargs):
        df = dm_den.load_data('dm_stats_20221208.h5')
        df.loc['mw', 'v_dot_phihat_disc(T<=1e4)'] = vc
        v0 = results['d'] * (vc / 100.) ** results['e']
        vdamp = results['h'] * (vc / 100.) ** results['j']
        ps = smooth_step_max(vs, v0, vdamp, results['k'])
        ax.plot(vs, ps, label='prediction from $v_\mathrm{c}$',
                #label = '$v_\mathrm{{c}} = {0:0.0f}\,\mathrm{{km\,s^{{-1}}}}$'\
                #        .format(vc),
                **kwargs)
        lowers, uppers = gal_bands('mw', vs, df, results, ddfrac, dhfrac, 
                                   ax=None)
        ax.fill_between(vs, lowers, uppers, 
                        alpha=0.9, 
                        color='#c0c0c0',
                        zorder=1, 
                        label='$1\sigma$ band')
        return None
    
    fig = plt.figure(figsize = (4.6 / 2. + 1., 2.5), dpi=600,
                     facecolor = (1., 1., 1., 0.))
    #fig = plt.figure(figsize = (5., 2.5), dpi=200)
    ax = fig.add_subplot(111)

    #predict(228., ax, c='C0', lw=3., dashes=[2., 0.5])
    predict(vc, ax, c='C3')

    ax.set_ylabel('$f(v)\,4\pi v^2\ [\mathrm{km^{-1}\,s}]$')
    ax.set_xlabel('$v\ [\mathrm{km\,s^{-1}}]$')
    ax.set_ylim(0., None)
    loc = [0.97,0.96]
    kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                      va='top', ha='right',
                      bbox=dict(facecolor='white', alpha=0.8, 
                                edgecolor='none'))
    ax.annotate('Milky Way', loc,
                **kwargs_txt)
    loc[1] -= 0.15
    kwargs_txt['fontsize'] = 11.
    ax.annotate('$v_\mathrm{{c}}={0:0.0f}\,\mathrm{{km\,s^{{-1}}}}$'
                .format(vc),
                loc, **kwargs_txt)

    # Put y-axis in scientific notation
    order_of_mag = -3
    ax.ticklabel_format(style='sci', axis='y', 
                        scilimits=(order_of_mag,
                                   order_of_mag),
                            useMathText=True)
    ax.legend(bbox_to_anchor=(0.5, -0.1), 
              loc='upper center', ncol=1,
              bbox_transform=fig.transFigure)
    #ax.legend(bbox_to_anchor=(0., -0.09), 
    #          loc='upper left', ncol=2,
    #          bbox_transform=fig.transFigure)

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=250)

    plt.show()

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

def calc_gs(vmins, vcircs, d, e, h, j, k, parallel=False):
    '''
    Calculate the value of the halo integral given vmin and circular
    velocity
    '''
    assert len(vmins) == len(vcircs)
    def normalize(vc):
        v0 = d * (vc / 100.) ** e
        vdamp = h * (vc / 100.) ** j
        N = scipy.integrate.quad(pN_smooth_step_max, 0., np.inf,
                                 (v0, vdamp, k), epsabs=0)[0]
        return N
    vcircs_set = np.array(list(set(vcircs)))
    N_dict = {vc: normalize(vc) for vc in vcircs_set}
    if parallel:
        print('in parallel')
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
    import dm_den
    df = dm_den.load_data('dm_stats_20221208.h5')
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
                              opts = {'parallel': parallel},
                              #nan_policy = 'omit')
                              nan_policy = 'propagate'
                              )
    params = model.make_params()
    params['d'].set(value=138.767313, vary=True, min=20., max=600.)
    params['e'].set(value=0.78734935, vary=True, min=0.01)
    params['h'].set(value=246.750219, vary=True, min=20., max=750.)
    params['j'].set(value=0.68338094, vary=True, min=0.01)
    params['k'].set(value=0.03089876, vary=True, min=0.001, max=1.)
    result = model.fit(np.log10(gs_truth), params, vmins=vs_truth, 
                       vcircs=vcircs)

    return result 
###############################################################################

def plt_universal(gals='discs', update_values=False,
                  tgt_fname=None, method='leastsq', 
                  vc100=True, err_method='sampling', ddfrac=None, dhfrac=None,
                  assume_corr=False,
                  band_alpha=0.4, data_color='grey', band_color='grey',
                  samples_color=plt.cm.viridis(0.5), ymax=None,
                  **kwargs):
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
    import dm_den
    import dm_den_viz
    df = dm_den.load_data('dm_stats_20221208.h5').drop(['m12w', 'm12z'])
    if gals != 'discs' and not isinstance(gals, (list, np.ndarray)):
        raise ValueError('Unexpected value provided for gals arg')
    if gals == 'discs':
        Ngals = 12
    else:
        Ngals = len(gals)
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
        ps = [smooth_step_max(v,
                              d * (vc) ** e,
                              h * (vc) ** j,
                              k)
              for v, vc in zip(vs, vcircs)]
        ps = np.array(ps)
        return ps
    
    model = lmfit.model.Model(calc_p,
                              independent_vars=['vs', 'vcircs'])
    params = model.make_params()

    if vc100:
        params['d'].set(value=114.970072, vary=True, min=0.)
        params['e'].set(value=0.92818194, vary=False, min=0.)
        params['h'].set(value=388.227498, vary=True, min=0.)
        params['j'].set(value=0.27035486, vary=False, min=0.)
        params['k'].set(value=0.03089876, vary=False, min=0.0001, max=1.)
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
        # Save raw variables to data_raw.pkl
        data2save = {key: result.params[key].value
                     for key in result.params.keys()}
        stderrs = {key+'_stderr': result.params[key].stderr
                   for key in result.params.keys()}
        covar = {'covar': result.covar}
        data2save = data2save | stderrs | covar #combine dictionaries
        dm_den.save_var_raw(data2save)

        p = result.covar.shape[0] #Number of parameters we're estimating
        N = 600 #There's 600 data points (50 bins for each disc) 
        z = 1. #Number of std deviations in forecast_siga
        #Probability of being within z std devs:
        P = scipy.special.erf(z / np.sqrt(2)) 
        forecast_sig = 1. - P

        # critical 2-tailed t value  
        tc = scipy.stats.t.ppf(q=1.-forecast_sig/2., df=N-p)

        for key in result.params.keys():
            if result.params[key].vary: 
                # Save strings to be used in paper.tex
                y = result.params[key].value
                stderr = result.params[key].stderr
                if key == 'd':
                    dy = ddfrac * y
                elif key == 'h':
                    dy = dhfrac * y
                else:
                    dy = stderr * tc
                # y_txt is a string. DY_TXT is an array of strings, or just an
                # array of just one string. Either way, 
                # type(DY_TXT) == np.ndarray, which is why we're denoting it in
                # all caps.
                y_txt, DY_TXT = staudt_utils.sig_figs(y, dy)
                dm_den.save_prediction(key, y_txt,  DY_TXT)
        vc_mw = dm_den_viz.vc_eilers 

        v0_mw = result.params['d'] * (vc_mw / 100.) ** result.params['e']
        dv0_mw = ddfrac * v0_mw
        v0_mw_txt, DV0_MW_TXT = staudt_utils.sig_figs(v0_mw, dv0_mw)
        dm_den.save_prediction('v0_mw', v0_mw_txt, DV0_MW_TXT)

        vdamp_mw = result.params['h'] * (vc_mw / 100.) ** result.params['j']
        dvdamp_mw = dhfrac * vdamp_mw
        vdamp_mw_txt, DVDAMP_MW_TXT = staudt_utils.sig_figs(vdamp_mw, 
                                                            dvdamp_mw)
        dm_den.save_prediction('vdamp_mw', vdamp_mw_txt, DVDAMP_MW_TXT)
    ###########################################################################

    if err_method == 'std_err':
        # Calculate the std error of the regression
        # p = 4  degrees of freedom negated by estimating d, e, h, and j:
        p = result.nvarys 
        # Std err of the regression
        s = np.sqrt(np.sum(result.residual ** 2.) / (result.ndata - p)) 

        N_vc = len(pdfs)
        vc_mean = df['v_dot_phihat_disc(T<=1e4)'].mean()
        var_vc = np.sum((df['v_dot_phihat_disc(T<=1e4)'] \
                            - vc_mean)**2.) / N_vc
        var_vs = np.sum((vs_truth.flatten() - vs_truth.mean())**2.) / result.ndata

    fig, axs = setup_universal_fig(gals)

    if isinstance(gals, (list, np.ndarray)):
        df = df.loc[gals]

    rms_dict = {}
    for i, gal in enumerate(df.index):
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        vcircs_postfit = np.repeat(vc, N_postfit)
        ps_postfit = result.eval(vs=vs_postfit, vcircs=vcircs_postfit)

        rms_err = calc_rms_err(pdfs[gal]['vs'], pdfs[gal]['ps'], 
                               calc_p,  
                               [np.repeat(vc, len(pdfs[gal]['vs'])), 
                                *[result.params[key] for key in ['d', 'e', 
                                                                 'h', 'j', 
                                                                 'k']]])
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
            axs[i+2].set_ylim(-resids_lim, resids_lim)
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
    if update_values:
        save_rms_errs({'universal': rms_dict})
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

    label_axes(axs, gals)

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=140)

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

    return result

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
    
    df = dm_den.load_data('dm_stats_20221208.h5').drop(['m12w', 'm12z'])
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
                 assume_corr=False):
    D_DEVS = np.random.normal(0., d * ddfrac, size=N)
    H_DEVS = np.random.normal(0., h * dhfrac, size=N)
    DEVS_UNCORR = np.array([D_DEVS, H_DEVS])
    MU = np.array([[d], [h]])
    if assume_corr:
        c = scipy.linalg.cholesky(covar, lower=True)
    else:
        c = np.identity(2)
    THETA = np.dot(c, DEVS_UNCORR) + MU

    D = THETA[0]
    V0HAT = D * (vc/100.) ** e

    H = THETA[1]
    VDAMP = H * (vc/100.) ** j

    ps_samples = np.array([smooth_step_max(vs,
                                           v0, vdamp, k)
                           for v0, vdamp in zip(V0HAT, VDAMP)])
    return ps_samples

def gal_bands(gal, vs, df, result, ddfrac=0.1, dhfrac=0.18, 
              assume_corr=False, ax=None, samples_color=plt.cm.viridis(0.5)):
    #dict_gal = pdfs[gal]
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
                              assume_corr=assume_corr)

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

    df = dm_den.load_data('dm_stats_20221208.h5')
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
            
    fig, axs = setup_universal_fig(gals)
    
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
    label_axes(axs, gals=gals)

    plt.show()

    return result

def three_d_distribs():
    import dm_den
    df = dm_den.load_data('dm_stats_20221208.h5').drop(['m12w', 'm12z'])
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
        df = dm_den.load_data('dm_stats_20221208.h5')

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
                     return_fracs=False):
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

def diff_fr68_agg(params, assume_corr=False, incl_area=True,
                  verbose=False):
    df = pd.read_pickle(paths.data + 'dm_stats_20221208.pkl')

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

def compare_methods(save_fname=None, verbose=False):
    import dm_den
    with open(paths.data + 'rms_errs.pkl', 'rb') as f:
        rms_dict = pickle.load(f)
    df = dm_den.load_data('dm_stats_20221208.h5')
    df_rms = pd.DataFrame.from_dict(rms_dict)
    df = pd.concat([df, df_rms], axis=1).drop(['m12w', 'm12z'])
    if verbose:
        df['diff'] = df['sigma_vc'] - df['universal']
        df.sort_values('diff', inplace=True)
        display(df[['diff']].T)

    fig = plt.figure(figsize=(4.7,3))
    ax = fig.add_subplot(111)
    df[['true_sigma', 
        #'sigma_vc', 
        'universal']].plot.bar(ax=ax, 
                               color=['C0', 
                                      #'orange', 
                                      'C3'],
                               width=0.7)
    tick_labels = ax.xaxis.get_majorticklabels()
    ax.set_xticklabels(tick_labels, rotation=40, ha='right', 
                       rotation_mode='anchor')
    ax.set_ylabel('RMS error $[\mathrm{km^{-1}\,s}]$')
    # Put y-axis in scientific notation
    order_of_mag = -4
    ax.ticklabel_format(style='sci', axis='y', 
                        scilimits=(order_of_mag,
                                   order_of_mag),
                        useMathText=True)
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    #labels[0] = '$\sigma(v_\mathrm{c})$'
    labels[0] = '$v_0=\sqrt{2/3}\sigma_\mathrm{meas}$'
    labels[1] = 'prediction from $v_\mathrm{c}$'
    ax.legend(labels=labels, bbox_transform=fig.transFigure, 
              bbox_to_anchor=(0.5, -0.0), loc='upper center', 
              ncol=3, fontsize=13)

    if save_fname is not None:
        plt.savefig(paths.figures + save_fname, bbox_inches='tight', dpi=140)
    plt.show()

    return None 
