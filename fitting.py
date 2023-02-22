import scipy
import warnings
import lmfit
import pickle
import math
import paths
import staudt_utils
import copy
import numpy as np
import pandas as pd
from progressbar import ProgressBar
import matplotlib as mpl
from matplotlib import pyplot as plt

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

def smooth_step_max(v, v0, vesc, k):
    '''
    Smooth-step-truncated Maxwellian, as opposed to the immediate cutoff
    of a Heaviside function used in trunc_max
    
    k is the strength of the exponential cutoff
    '''
    
    def calc_pN(v, v0, vesc):
        '''
        Probability density before normalizing by N
        '''
        fN = np.exp( - v**2. / v0**2. )
        pN = fN * 4. * np.pi * v**2.
        trunc = 1. / (1. + np.exp(-k * (vesc-v)))
        pN *= trunc
        return pN
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        N = scipy.integrate.quad(calc_pN, 0., np.inf, (v0, vesc), epsabs=0)[0]
        p = calc_pN(v, v0, vesc) / N
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
        p.add('vesc', value=470., vary=True, min=250., max=600.)
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

def plt_naive(gals='discs', tgt_fname=None):
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
    xfigsize = 4.5 * Ncols + 1.
    yfigsize = 3.7 * Nrows + 1. 
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

    # velocities to use when plotting the functional form attempts
    vs_maxwell = np.linspace(0., 800., 700) 
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        # Strength at with to truncate the distribution
        # I'm probably not going to use this here, though
        k = pickle.load(f)['k'] 

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
        ps_sigma_vc = smooth_step_max(
                                   vs_maxwell, 
                                   np.sqrt(2./3.)*sigma_predicted,
                                   np.inf,
                                   np.inf)
        ps_true_sigma = smooth_step_max(vs_maxwell,
                                        np.sqrt(2./3.) * sigma_truth,
                                        np.inf,
                                        np.inf)
        axs[i].stairs(ps_truth, bins,
                      color='grey')
        axs[i].plot(vs_maxwell, ps_sigma_vc, 
                    label='$v_0=\sqrt{2/3}\sigma_\mathrm{3D}(v_\mathrm{c})$'
                   )
        axs[i].plot(vs_maxwell, ps_true_sigma,
                    label = '$v_0=\sqrt{2/3}\sigma_\mathrm{3D,measured}$')

        # Draw vesc line
        axs[i].axvline(vesc, ls='--', alpha=0.5, color='grey')
        trans = mpl.transforms.blended_transform_factory(axs[i].transData,
                                                         axs[i].transAxes)
        axs[i].text(vesc, 0.5, '$v_\mathrm{esc}$', transform=trans,
                    fontsize=15., rotation=90., color='gray', 
                    horizontalalignment='right')

        axs[i].grid(False)
        axs[i].tick_params(axis='x', direction='inout', length=6)

        loc=[0.97,0.96]
        kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                      va='top', ha='right', 
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        axs[i].annotate(gal, loc,
                        **kwargs_txt)
        '''
        loc[1] -= 0.1
        kwargs_txt['fontsize'] = 11.
        axs[i].annotate('$v_\mathrm{{esc}}={0:0.0f}$\n'
                        '$v_\mathrm{{damp}}={1:0.0f}$\n'
                        '$\chi^2={2:0.3e}$'
                        .format(vesc,
                                result.params['vdamp'].value,
                                result.chisqr
                               ),
                        loc, **kwargs_txt)
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
                resids_extend = smooth_step_max(
                    vs_extend,
                    np.sqrt(2./3.) * sigma,
                    np.inf, np.inf)
                resids = np.append(resids, 
                                   resids_extend,
                                   axis=0)
                return resids

            axs[i+2].grid(False)
            axs[i+2].plot(vs_resids, calc_resids(sigma_predicted))
            axs[i+2].plot(vs_resids, calc_resids(sigma_truth))

            loc[1] -= 0.1
            kwargs_txt['fontsize'] = 11.
            axs[i].annotate(
                            '$\mathrm{SSE}={2:0.3e}$'
                            .format(
                                     
                                   ),
                            loc, **kwargs_txt)
            if i == 0:
                axs[i+2].set_ylabel('resids')
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.04),
                  bbox_transform=fig.transFigure, ncol=3)
    label_axes(axs, gals)

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=140)

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
    fig.Nrows = Nrows

    return fig, axs

def plt_universal(gals='discs', update_values=False,
                  tgt_fname=None, method='leastsq', 
                  vc100=True, err_method='sampling', ddfrac=None, dhfrac=None,
                  band_alpha=0.4, data_color='grey', band_color='grey',
                  **kwargs):
    if (ddfrac is not None or dhfrac is not None) and err_method != 'sampling':
        raise ValueError('ddfrac and dhfrac are only used in sampling.')
    elif err_method == 'sampling':
        if ddfrac is None:
            ddfrac = 0.1
        if dhfrac is None:
            dhfrac = 0.18
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
                dy = stderr * tc
                # y_str is a string. dy_str is an array or strings (or just an
                # array of just one string).
                y_str, dy_str = staudt_utils.sig_figs(y, dy)
                dm_den.save_prediction(key, y_str,  dy_str)
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

    for i, gal in enumerate(df.index):
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        vcircs_postfit = np.repeat(vc, N_postfit)

        ps_postfit = result.eval(vs=vs_postfit, vcircs=vcircs_postfit)

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
            lowers, uppers = gal_bands(gal, vs_postfit, pdfs, df, 
                                       result, ddfrac=ddfrac, dhfrac=dhfrac,
                                       ax = axs[i])

            axs[i].fill_between(vs_postfit, lowers, uppers, color=band_color, 
                                alpha=band_alpha, 
                                ec=None, zorder=1)

        # Plot data
        axs[i].stairs(pdfs[gal]['ps'], pdfs[gal]['bins'], color=data_color,
                      label='data')

        # Plot prediction
        axs[i].plot(vs_postfit,
                    ps_postfit,
                    '-',
                    label='prediction', color='b', lw=1.)

        loc = [0.97,0.96]
        kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                          va='top', ha='right',
                          bbox=dict(facecolor='white', alpha=0.8, 
                                    edgecolor='none'))
        axs[i].annotate(gal, loc,
                        **kwargs_txt)
        loc[1] -= 0.08
        kwargs_txt['fontsize'] = 13.
        axs[i].annotate('$v_\mathrm{{c}}={0:0.0f}\,\mathrm{{km\,s^{{-1}}}}$'
                        .format(vc), loc, **kwargs_txt)

        axs[i].grid(False)
    if fig.Nrows == 3:
        legend_y = 0.05
    else:
        legend_y = -0.04
    axs[0].legend(
                  bbox_to_anchor=(0.5, legend_y), 
                  loc='upper center', ncol=4,
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

def find_uncertainty(gals, ddfrac=0.1, dhfrac=0.18, v0char=1., N_samples=1000):
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
                              ddfrac, dhfrac)

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

def make_samples(N, vs, vc, d, e, h, j, k, covar, ddfrac, dhfrac):
    D_DEVS = np.random.normal(0., d * ddfrac, size=N)
    H_DEVS = np.random.normal(0., h * dhfrac, size=N)
    DEVS_UNCORR = np.array([D_DEVS, H_DEVS])
    MU = np.array([[d], [h]])
    c = scipy.linalg.cholesky(covar, lower=True)
    THETA = np.dot(c, DEVS_UNCORR) + MU

    D = THETA[0]
    V0HAT = D * (vc/100.) ** e

    H = THETA[1]
    VDAMP = H * (vc/100.) ** j

    ps_samples = np.array([smooth_step_max(vs,
                                           v0, vdamp, k)
                           for v0, vdamp in zip(V0HAT, VDAMP)])
    return ps_samples

def gal_bands(gal, vs, pdfs, df, result, ddfrac=0.1, dhfrac=0.18, ax=None):
    dict_gal = pdfs[gal]

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

    N_samples = 1000
    ps_samples = make_samples(N_samples, vs, vc, 
                              d, e, h, j, k, covar, ddfrac, dhfrac)

    P_1std = scipy.special.erf(1. / np.sqrt(2)) # ~68%
    lower_q = (1. - P_1std) / 2. 
    upper_q = lower_q + P_1std
    lower_q *= 100.
    upper_q *= 100.
    lowers = np.percentile(ps_samples, lower_q, axis=0)
    uppers = np.percentile(ps_samples, upper_q, axis=0)

    if ax is not None:
        for ps in ps_samples:
            ax.plot(vs, ps, color='g', alpha=0.1, zorder=0)

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

def diff_fr68(params, pdfs, df):
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        data_raw = pickle.load(f)

    def count_within(gal, ddfrac=0.1, dhfrac=0.18):
        pdf_gal = pdfs[gal]
        ps = pdf_gal['ps']
        vs = pdf_gal['vs']
        lowers, uppers = gal_bands(gal, vs, pdfs, df, data_raw, ddfrac, 
                                   dhfrac)
        is_above = ps > uppers
        is_below = ps < lowers
        is_outlier = is_above | is_below 
        N_out = np.sum(is_outlier)
        N_tot = len(vs)
        
        percent = 1. - N_out / N_tot
        return percent

    ddfrac = params['ddfrac']
    dhfrac = params['dhfrac']
    percents_within = np.array([count_within(gal, ddfrac, dhfrac) \
                                for gal in pdfs])
    P_1std = scipy.special.erf(1. / np.sqrt(2.)) # ~68%
    resids_vals = percents_within - P_1std 
    print(ddfrac.value, dhfrac.value, np.sum(resids_vals ** 2.))
    return resids_vals

def diff_fr68_agg(params, pdfs, df):
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        data_raw = pickle.load(f)

    def count_within(ddfrac=0.1, dhfrac=0.18):
        N_out = 0.
        N_tot = 0.

        for gal in pdfs:
            pdf_gal = pdfs[gal]
            ps = pdf_gal['ps']
            vs = pdf_gal['vs']
            lowers, uppers = gal_bands(gal, vs, pdfs, df, data_raw, ddfrac, 
                                       dhfrac)
            is_above = ps > uppers
            is_below = ps < lowers
            is_outlier = is_above | is_below 
            N_out += np.sum(is_outlier)
            N_tot += len(vs)
        
        percent = 1. - N_out / N_tot
        return percent

    ddfrac = params['ddfrac']
    dhfrac = params['dhfrac']
    percents_within = np.array([count_within(gal, ddfrac, dhfrac) \
                                for gal in pdfs])
    P_1std = scipy.special.erf(1. / np.sqrt(2.)) # ~68%
    resids_vals = percents_within - P_1std 
    print(ddfrac.value, dhfrac.value, np.sum(resids_vals ** 2.))
    return resids_vals

def find_68_uncertainty(method, **kwargs):
    import dm_den

    df = dm_den.load_data('dm_stats_20221208.h5').drop(['m12w', 'm12z'])
    pdfs = copy.deepcopy(pdfs_v)
    del pdfs['m12w']
    del pdfs['m12z']

    params = lmfit.Parameters()
    params.add('ddfrac', value=0.033, vary=True, min=0.)
    params.add('dhfrac', value=0.296, vary=True, min=0.)

    minimizer_result = lmfit.minimize(diff_fr68, params, method=method,
                                      args=(pdfs, df), **kwargs)
    display(minimizer_result)

    return None
