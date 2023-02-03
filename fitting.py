import scipy
import warnings
import lmfit
import pickle
import math
import paths
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
        axs[i].grid()
        
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

    # velocities to use when plotting the functional form attempts
    vs_maxwell = np.linspace(0., 800., 700) 
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        # Strength at with to truncate the distribution
        # I'm probably not going to use this here, though
        k = pickle.load(f)['k'] 

    def resids(p, sigma, vs_truth, ps_truth):
        k = p['k'].value
        vdamp = p['vdamp'].value
        ps_predicted = smooth_step_max(
                                   vs_truth, 
                                   np.sqrt(2./3.)*sigma,
                                   vdamp,
                                   k)
        resids = ps_predicted - ps_truth
        return resids

    for i, gal in enumerate(df.index):
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        with open(paths.data + 'data_raw.pkl', 'rb') as f:
            results_dict = pickle.load(f)
        sigma_predicted = 10.**results_dict['disp_amp'] \
                          * vc ** results_dict['disp_slope']
        sigma_truth = df.loc[gal, 'disp_dm_disc_cyl']
        vesc = vesc_dict[gal]['ve_avg']
        bins = pdfs_v[gal]['bins']
        vs_truth = (bins[1:] + bins[:-1]) / 2.
        ps_truth = pdfs_v[gal]['ps']
        ps_sigma = smooth_step_max(vs_maxwell, 
                                   np.sqrt(2./3.)*sigma_predicted,
                                   np.inf,
                                   np.inf)

        axs[i].stairs(ps_truth, bins,
                      color='grey')
        axs[i].plot(vs_maxwell, ps_sigma, 
                    label='$v_0=\sqrt{2/3}\sigma(v_\mathrm{c})$, not truncated'
                   )
        axs[i].plot(vs_maxwell, smooth_step_max(vs_maxwell,
                                                vc,
                                                np.inf,
                                                np.inf),
                    label='$v_0=v_\mathrm{c}$, not truncated')


        '''
        p = lmfit.Parameters()
        p.add('k', value=k, vary=False)
        p.add('vdamp', value=450., vary=True, min=200., max=800.)
        result = lmfit.minimize(resids, p, args=(sigma_predicted,
                                                 vs_truth,
                                                 ps_truth),
                                method='nelder')
        axs[i].plot(vs_maxwell, smooth_step_max(vs_maxwell, 
                                                np.sqrt(2./3.) \
                                                    * sigma_predicted,
                                                result.params['vdamp'].value,
                                                k))
        '''
        '''
        axs[i].plot(vs_maxwell, smooth_step_max(vs_maxwell,
                                               df.loc[gal, 'v0'],
                                               np.inf,
                                               np.inf))
        '''

        #axs[i].axvline(result.params['vdamp'].value, ls='--', alpha=0.5)

        # Draw vesc line
        axs[i].axvline(vesc, ls='--', alpha=0.5, color='grey')
        trans = mpl.transforms.blended_transform_factory(axs[i].transData,
                                                         axs[i].transAxes)
        axs[i].text(vesc, 0.5, '$v_\mathrm{esc}$', transform=trans,
                    fontsize=15., rotation=90., color='gray', 
                    horizontalalignment='right')

        axs[i].grid()

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
    if tgt_fname is not None and not plot:
        raise ValueError('tgt_fname for plot image can only be specified if'
                         ' plot is True')
    import dm_den
    df = dm_den.load_data('dm_stats_20221208.h5').drop(['m12w', 'm12z'])
    if gals != 'discs' and not isinstance(gals, (list, np.ndarray)):
        raise ValueError('Unexpected value provided for gals arg')
    pdfs = pdfs_v.copy()
    pdfs.pop('m12z')
    pdfs.pop('m12w')

    for gal in pdfs:
        dict_gal = pdfs[gal]
        bins = dict_gal['bins']
        dict_gal['vs'] = (bins[1:] + bins[:-1]) / 2.
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

def plt_universal(gals='discs', update_value=False,
                  tgt_fname=None, method='leastsq', 
                  vc100=True, **kwargs):
    import dm_den
    df = dm_den.load_data('dm_stats_20221208.h5').drop(['m12w', 'm12z'])
    if gals != 'discs' and not isinstance(gals, (list, np.ndarray)):
        raise ValueError('Unexpected value provided for gals arg')
    pdfs = pdfs_v.copy()
    pdfs.pop('m12z')
    pdfs.pop('m12w')

    for gal in pdfs:
        dict_gal = pdfs[gal]
        bins = dict_gal['bins']
        dict_gal['vs'] = (bins[1:] + bins[:-1]) / 2.
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
    vs_postfit = np.linspace(0., 750., N_postfit)

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
        params['e'].set(value=0.92818194, vary=True, min=0.)
        params['h'].set(value=388.227498, vary=True, min=0.)
        params['j'].set(value=0.27035486, vary=True, min=0.)
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

    data2save = {key: result.params[key].value
                 for key in result.params.keys()}
    stderrs = {key+'_stderr': result.params[key].stderr
               for key in result.params.keys()}
    covar = {'covar': result.covar}
    data2save = data2save | stderrs | covar #combine dictionaries
    dm_den.save_var_raw(data2save)
    ###########################################################################

    fig, axs = setup_universal_fig(gals)

    if isinstance(gals, (list, np.ndarray)):
        df = df.loc[gals]

    for i, gal in enumerate(df.index):
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        vcircs_postfit = np.repeat(vc, N_postfit)

        ps_postfit = result.eval(vs=vs_postfit, vcircs=vcircs_postfit)

        try:
            # Plot 3sigma band
            result.eval_uncertainty(sigma=3, vs=vs_postfit, vcircs=vcircs_postfit)
            axs[i].fill_between(vs_postfit,
                                ps_postfit-result.dely,
                                ps_postfit+result.dely,
                                color='grey', alpha=0.4, ec=None,
                                label='$3\sigma$ band')

            # Plot 1sigma band
            result.eval_uncertainty(sigma=1, vs=vs_postfit, vcircs=vcircs_postfit)
            axs[i].fill_between(vs_postfit,
                                ps_postfit-result.dely,
                                ps_postfit+result.dely,
                                color='blue', alpha=0.8, ec=None,
                                label='$1\sigma$ band')
        except AttributeError as error:
            print(error)

        # Plot data
        axs[i].stairs(pdfs[gal]['ps'], pdfs[gal]['bins'], color='grey',
                      label='data')

        # Plot prediction
        axs[i].plot(vs_postfit,
                    ps_postfit,
                    '-',
                    label='prediction', color='r', lw=1.)

        loc = [0.97,0.96]
        kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                          va='top', ha='right',
                          bbox=dict(facecolor='white', alpha=0.8, 
                                    edgecolor='none'))
        axs[i].annotate(gal, loc,
                        **kwargs_txt)

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
        # otherwise we'll lock in the upper y limit for each row with the
        # automatically determined upper limit for the galaxy in the first
        # column of that row.
        axs[i].set_ylim(bottom=0.)
    plt.show()

    return result

def plt_universal_mc(gals='discs'):
    '''
    Plot uncertainty in distribution with a monte carlo method
    '''
    import dm_den
    df = dm_den.load_data('dm_stats_20221208.h5')
    df.drop(['m12z', 'm12w'], inplace=True)

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
    z = 1. #Number of std deviations in forecast_siga
    #Probability of being within z std devs:
    P = scipy.special.erf(z / np.sqrt(2)) 
    forecast_sig = 1. - P

    # critical 2-tailed t value  
    tc = scipy.stats.t.ppf(q=1.-forecast_sig/2., df=N-p)

    vs = np.linspace(0., 675., 1000)

    d_best = params['d'].value
    e_best = params['e'].value
    h_best = params['h'].value
    j_best = params['j'].value
    k = params['k'].value

    # c is a matrix for which c*c^T = covar
    c = scipy.linalg.cholesky(covar, lower=True)

    N = 1000
    d_z_samples = np.random.normal(0., result.params['d'].stderr * tc, size=N)
    e_z_samples = np.random.normal(0., result.params['e'].stderr * tc, size=N)
    h_z_samples = np.random.normal(0., result.params['h'].stderr * tc, size=N)
    j_z_samples = np.random.normal(0., result.params['j'].stderr * tc, size=N)
    k_z_samples = np.random.normal(0., result.params['k'].stderr * tc, size=N)

    assert list(vary_dict.keys()) == ['d', 'e', 'h', 'j', 'k']
    mu_theta = np.array([d_best, e_best, h_best, j_best, k])
    mu_theta = mu_theta[vary_mask] #Only use parameters we're varying
    mu_theta = mu_theta.reshape((p, 1))
    # Matrix of uncorrelated z-scores:
    Z_uncorr = np.array([d_z_samples, e_z_samples, h_z_samples, 
                         j_z_samples, k_z_samples])
    Z_uncorr = Z_uncorr[vary_mask] #Only use parameters we're varying
    # Apply correlation to parameter samples
    theta = np.dot(c, Z_uncorr) + mu_theta 

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

    for i, gal in enumerate(df.index):
        ax = axs[i]
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']

        v0 = d * vc ** e 
        vdamp = h * vc ** j 

        for l in range(N): 
            ax.plot(vs, smooth_step_max(vs, v0[l], vdamp[l], k), 
                    c='b', alpha=0.05)

        ax.stairs(pdfs_v[gal]['ps'], pdfs_v[gal]['bins'], color='grey')
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
    pdfs = pdfs_v.copy()
    pdfs.pop('m12z')
    pdfs.pop('m12w')

    for gal in pdfs:
        dict_gal = pdfs[gal]
        bins = dict_gal['bins']
        dict_gal['vs'] = (bins[1:] + bins[:-1]) / 2.
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
    
    '''
    # Plot the data
    fig, ax = plt.subplots(figsize=(6,4), subplot_kw={"projection": "3d"},
                           dpi=180)
    ax.plot_wireframe(vs_truth, vcircs, 
                      ps_truth_smooth)
    plt.show()
    '''

    vs_postfit = np.linspace(0., 700., 1000)

    ###########################################################################
    # Trying to smooth the data plot
    ###########################################################################
    vcircs_set = df['v_dot_phihat_disc(T<=1e4)'].values
    vcircs_postfit = np.linspace(vcircs_set.min(), vcircs_set.max(), 1000)
    X, Y = np.meshgrid(vs_postfit, vcircs_postfit)

    m = 600.
    s = np.mean([m-np.sqrt(2.*m), 
                 m+np.sqrt(2.*m)])
    
    data = np.array([vs_truth, vcircs, ps_truth])
    filtered = scipy.ndimage.gaussian_filter(data, sigma=[0., 0.8, 0.])
    
    '''
    fig = plt.figure(dpi=200, figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(*filtered, cmap=mpl.cm.coolwarm)
    plt.show()
    '''

    ps_truth_smooth = scipy.interpolate.griddata((vs_truth.flatten(), 
                                                  vcircs.flatten()), 
                                                 ps_truth.flatten(),
                                                 (vs_postfit[None,:], 
                                                  vcircs_postfit[:,None]),
                                                 method='linear')
    interped = np.array([X, Y, ps_truth_smooth])
    filtered_interp = scipy.ndimage.gaussian_filter(interped, 
                                                    sigma=[0., 25., 0.7])

    fig = plt.figure(dpi=190, figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(*filtered_interp, 
                    cmap=mpl.cm.coolwarm, rcount=100,
                    ccount=300, antialiased=False)
    ax.view_init(elev=33., azim=-102.)
    ax.set_ylabel('\n$v_\mathrm{c}\ \mathrm{\left[km\,s^{-1}\\right]}$')
    ax.set_xlabel('\n$v\ \mathrm{\left[km\,s^{-1}\\right]}$', y=-100)
    ax.set_zlabel('\n$f(v)\,4\pi v^2\ \mathrm{\left[km^{-1}\,s\\right]}$',
                  linespacing=3.)
    plt.show()
    ###########################################################################

    X, Y = np.meshgrid(vs_postfit, df['v_dot_phihat_disc(T<=1e4)'].values)
    # Generate the prediction
    vcircs = df['v_dot_phihat_disc(T<=1e4)'].values
    vcircs = np.sort(vcircs)
    X, Y = np.meshgrid(vs_postfit, vcircs)
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        params = pickle.load(f)
    V0 = params['d'] * Y/100. ** params['e']
    Vdamp = params['h'] * Y/100. **params['j']
    Z = [[smooth_step_max(v, 
                          params['d'] * (vc/100.) ** params['e'],
                          params['h'] * (vc/100.) ** params['j'],
                          params['k']) 
          for v in vs_postfit]
         for vc in vcircs]
    Z = np.array(Z)

    # Plot the prediction
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_wireframe(X, Y, Z)
    plt.show()

    fig = plt.figure(dpi=190, figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z,
                    cmap=mpl.cm.coolwarm, rcount=100,
                    ccount=300, antialiased=False)
    ax.view_init(elev=33., azim=-102.)
    ax.set_ylabel('\n$v_\mathrm{c}\ \mathrm{\left[km\,s^{-1}\\right]}$')
    ax.set_xlabel('\n$v\ \mathrm{\left[km\,s^{-1}\\right]}$', y=-100)
    ax.set_zlabel('\n$f(v)\,4\pi v^2\ \mathrm{\left[km^{-1}\,s\\right]}$',
                  linespacing=3.)
    plt.show()

    # Generate the prediction
    vs_postfit = np.linspace(0., 650., 100)
    vcircs_set = df['v_dot_phihat_disc(T<=1e4)'].values
    vcircs_postfit = np.linspace(vcircs_set.min(), vcircs_set.max(), 100)
    #X, Y = np.meshgrid(vs_postfit, vcircs_postfit)

    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        params = pickle.load(f)
    V0 = params['d'] * Y/100. ** params['e']
    Vdamp = params['h'] * Y/100. **params['j']
    '''
    Z = [[smooth_step_max(v, 
                          params['d'] * (vc/100.) ** params['e'],
                          params['h'] * (vc/100.) ** params['j'],
                          params['k']) 
          for v in vs_postfit]
         for vc in vcircs_postfit]
    Z = np.array(Z)
    '''
    vs_postfit, vcircs_postfit = np.meshgrid(vs_postfit, vcircs_postfit)
    vs_postfit = vs_postfit.flatten()
    vcircs_postfit = vcircs_postfit.flatten()
    v0s = params['d'] * (vcircs_postfit/100.) ** params['e']
    vdamps = params['h'] * (vcircs_postfit/100.) ** params['j']
    zs = [smooth_step_max(v, v0, vdamp, params['k'])
          for v, v0, vdamp in zip(vs_postfit, v0s, vdamps)]
    zs = np.array(zs)
    print(zs)

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
    fig = plt.figure(dpi=190, figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z_smooth,
                    cmap=mpl.cm.coolwarm, rcount=100,
                    ccount=300, antialiased=False)
    ax.view_init(elev=33., azim=-102.)
    ax.set_ylabel('\n$v_\mathrm{c}\ \mathrm{\left[km\,s^{-1}\\right]}$')
    ax.set_xlabel('\n$v\ \mathrm{\left[km\,s^{-1}\\right]}$', y=-100)
    ax.set_zlabel('\n$f(v)\,4\pi v^2\ \mathrm{\left[km^{-1}\,s\\right]}$',
                  linespacing=3.)
    plt.show()
