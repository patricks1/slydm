import scipy
import warnings
import lmfit
import pickle
import math
import paths
import numpy as np
from progressbar import ProgressBar
import matplotlib as mpl
from matplotlib import pyplot as plt

with open('./data/v_pdfs.pkl','rb') as f:
    pdfs_v=pickle.load(f)
vs_pdfs=np.array([pdfs_v[galname]['bins'] 
                  for galname in pdfs_v]).flatten()
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
    if gals == 'discs':
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

def plt_universal(gals='discs', method='leastsq', update_value=False,
                  tgt_fname=None, **kwargs):
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
    p.add('d', value=1.60030613, vary=True, min=0.1, max=4.)
    p.add('e', value=0.92819047, vary=True, min=0.1, max=4.)
    p.add('h', value=111.783463, vary=True, min=5., max=300.)
    p.add('j', value=0.27035357, vary=True, min=0.05, max=4.)
    p.add('k', value=0.03089489, vary=True, min=0.0001, max=1.)

    ############################################################################
    ## Fitting escape velocity
    ############################################################################
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
    data2save = {key: result_dehjk.params[key].value
                 for key in result_dehjk.params.keys()}
    dm_den.save_var_raw(data2save)

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
