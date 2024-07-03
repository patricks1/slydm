def read(samples_fname, consider_burnin=True, flat=True, verbose=False,
         make_thin=True):
    import emcee
    import paths
    import warnings
    import numpy as np

    reader = emcee.backends.HDFBackend(paths.data + samples_fname)
    if consider_burnin:
        try:
            tau = reader.get_autocorr_time()
            if make_thin:
                thin = int(0.5 * np.min(tau))
            else:
                thin = 1
        except emcee.autocorr.AutocorrError as err:
            tau = reader.get_autocorr_time(quiet=True)
            thin = 1
        burnin = int(2 * np.max(tau))
        samples = reader.get_chain(discard=burnin, thin=thin, flat=flat)
        if verbose:
            print('tau: {2}'
                  '\nburn-in: {0:0.0f}'
                  '\nthin: {1:0.0f}'
                  .format(burnin, thin, tau))
    else:
        samples = reader.get_chain(flat=flat)

    return samples

def plot_chains(samples_fname):
    import emcee
    import paths
    import matplotlib.pyplot as plt

    reader = emcee.backends.HDFBackend(paths.data + samples_fname)
    samples = reader.get_chain(flat=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(samples[:, 0], 'b.')
    plt.show()
    return None

def corner_plot(samples_fname, consider_burnin=True, log_prior_function=None,
                make_thin=True, rng=None):
    import corner
    import scipy
    import copy
    import emcee
    import paths
    import pickle
    import multiprocessing
    import matplotlib.pyplot as plt
    import numpy as np
    from progressbar import ProgressBar

    samples = read(samples_fname, consider_burnin=consider_burnin, flat=False,
                   verbose=True, make_thin=make_thin)
    print('samples shape: {0}'
          .format(str(samples.shape)))
    flat_samples = samples.reshape(-1, samples.shape[-1])
    ndim = flat_samples.shape[1]
    fig = corner.corner(flat_samples, labels=['D', 'e', 'H', 'j', 'k'],
                        range=rng)
    axs = np.array(fig.axes).reshape((ndim, ndim))

    if log_prior_function is not None:
        with open(paths.data + 'data_raw.pkl', 'rb') as f:
            ls_result = pickle.load(f)

            # Best estimate of the parameters from least squares minimization
            mu = np.array([ls_result[param] 
                           for param in ['d', 'e', 'h', 'j', 'k']])
        nwalkers = 32
        pos = mu + 1.e-4 * np.random.randn(nwalkers, ndim)
        with multiprocessing.Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prior_function,
                                            pool=pool)
            print('\nSampling prior')
            sampler.run_mcmc(pos, int(1e4), progress=True)
            tau = sampler.get_autocorr_time(quiet=True)
            burnin = int(2 * np.max(tau))
            samples = sampler.get_chain(flat=True, discard=burnin)
    
        ## For each plot on the diagonal, marginalize over all the other
        ## parameters.

        ## Initialize a placeholder for the ranges that the corner plot 
        ## currently shows for each of i
        ## the 
        ## parameters
        #xranges = np.zeros((ndim, 2))

        ## Initialize a placeholder for all parameter values that the corner 
        ## plot currently shows for each 
        ## of the 
        ## parameters
        #Xs = []

        #for i in range(ndim):
        #    # For each parameter, get the values that the corner plot
        #    # currently shows.
        #    ax = axs[i, i]
        #    patches = ax.patches
        #    polygon = patches[0]
        #    vertices = polygon.get_path().vertices
        #    xs = vertices[:, 0] # All theta[i] values
        #    Xs.append(xs)
        #    
        #    # Min and max theta[i] values
        #    xranges[i] = np.array([xs.min(), xs.max()]) 

        for i in range(ndim):
        #    def func(n0, n1, n2, n3, ti):
        #        '''
        #        Convert the log_prior_function so the nuicance parameters are
        #        the first four args and the parameter of interest is last so
        #        scipy.integrate.nquad can integrate over the nuicance
        #        parameters
        #        '''
        #        theta = np.array([n0, n1, n2, n3])
        #        theta = np.insert(theta, i, ti)
        #        return log_prior_function(theta)


            ax = axs[i, i]
        #    xs = Xs[i]
        #    logys = np.zeros(len(xs))

        #    # Our integration bounds correspond only to the nuicance parameters
        #    # i.e. not the i'th parameter
        #    nuicance_ranges = np.delete(xranges, i, axis=0)

        #    # Perform the marginalization.
        #    pbar = ProgressBar()
        #    for j, x in enumerate(pbar(xs)):
        #        logys[j] = scipy.integrate.nquad(
        #                func, nuicance_ranges, args=(x,)
        #        )[0]
        #    ys = np.exp(np.array(logys))

            
            ax_twin = ax.twinx()
            ax_twin.hist(samples[:, i], bins=150, density=True, 
                         histtype='step')

    plt.show()

    return None

def trace_plot(samples_fname):
    import matplotlib.pyplot as plt
    import numpy as np

    samples = read(samples_fname, consider_burnin=True, flat=False,
                   verbose=True)
    print('samples shape: {0}'
          .format(str(samples.shape)))
    flat_samples = samples.reshape(-1, samples.shape[-1])
    ndim = flat_samples.shape[1]
    
    trace = [flat_samples[:x].mean(axis=0) 
             for x in range(1, len(flat_samples))]
    trace = np.array(trace)

    labels = ['D', 'e', 'H', 'j', 'k']
    ndim = samples.shape[-1]
    fig, axs = plt.subplots(ndim, 2, figsize=(8, 12), sharey=False,
                            sharex='col')
    fig.subplots_adjust(wspace=.3, hspace=0.)
    for i in range(ndim):
        axs[i, 0].plot(trace[:, i])
        axs[i, 0].set_ylabel(labels[i])

        xs = range(int(5e3), len(trace))
        axs[i, 1].plot(xs, trace[int(5e3):, i])

    axs[0, 0].set_title('All samples')
    axs[0, 1].set_title('Starting @ 5000')

    plt.show()

    return None

def estimate(samples_fname, result_fname=None, update_paper=False):
    '''
    Read an emcee samples file, calculate the best estimates and errors on
    the parameters, display those estimates and errors, and return a dictionary
    of the
    best estimates. The best estimates are the medians. The positive and
    negative errors represent the distances of the 84th and 16th percentiles
    from the median.

    Parameters
    ----------
    samples_fname: str
        The name of the file where emcee stored its chains
    result_fname: str, default None
        Where to save the parameter estimates. It should have a '.pkl' 
        extension.

    Returns
    -------
    results_dict: dict
        The best estimates of the parameters from the chains.
    '''
    import dm_den
    import staudt_utils
    import UCI_tools.tools as uci
    import numpy as np
    from IPython.display import display, Math

    samples = read(samples_fname)
    labels = ['d', 'e', 'h', 'j', 'k']

    ndim = samples.shape[1]
    results_dict = {}
    for i in range(ndim):
        est = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(est)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.4f}}}^{{+{2:.4f}}}"
        txt = txt.format(est[1], q[0], q[1], labels[i])
        display(Math(txt))
        results_dict[labels[i]] = est[1]
        results_dict['d' + labels[i]] = np.array([q[0], q[1]])
    if result_fname is not None:
        dm_den.save_var_raw(results_dict, result_fname) 
    if update_paper:
        for key in labels:
            # Save strings to be used in paper.tex
            y = results_dict[key]
            DY = results_dict['d' + key]
            # y_txt is a string. DY_TXT is an array of strings, or just an
            # array of just one string. Either way, 
            # type(DY_TXT) == np.ndarray, which is why we're denoting it in
            # all caps.
            y_txt, DY_TXT = staudt_utils.sig_figs(y, DY)
            uci.save_prediction(key, y_txt,  DY_TXT)
    return results_dict 

def make_gal_distrib_samples(df, gal, THETA, vs):
    import fitting
    import numpy as np

    vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
    D, E, H, J, K = THETA.T
    V0 = D * (vc / 100.) ** E
    VDAMP = H * (vc / 100.) ** J
    ps_samples = np.array([fitting.smooth_step_max(vs,
                                                   v0,
                                                   vdamp, 
                                                   k)
                           for v0, vdamp, k in zip(V0, VDAMP, K)])
    return ps_samples

def make_distrib_samples(df_source, mcmc_samples_source, tgt_fname):
    import dm_den
    import h5py
    import paths
    import numpy as np
    from progressbar import ProgressBar

    flat_samples = read(mcmc_samples_source)
    indices = np.random.randint(len(flat_samples), size=5000)
    THETA = flat_samples[indices]

    nondisks = ['m12z', 'm12w']
    df = dm_den.load_data(df_source).drop(nondisks)

    vs = np.linspace(0., 700., 300)
    distrib_samples = {}
    distrib_samples['vs'] = vs

    with h5py.File(paths.data + tgt_fname, 'w') as f:
        f.create_dataset('vs', data=vs)
        pbar = ProgressBar()
        for galname in pbar(df.index):
            distrib_samples[galname] = make_gal_distrib_samples(
                    df, 
                    galname,
                    THETA,
                    vs
            )
            f.create_dataset(galname, data=distrib_samples[galname])
    return None

def make_distrib_samples_by_v0(
        df_source, 
        mcmc_samples_source, 
        maxv0=2.3,
        Nvs=30,
        tgt_fname_override=None):
    '''
    Make speed distributions so that each galaxy's speed distribution has the
    same vs_gal / v0_gal, meaning vs_gal will be different for each galaxy.
    Save the results in an HDF5 file at `paths.data + tgt_fname`. The 
    structure of the h5py.File `f` is as follows:
        f[galname]['vs']: The speeds at which the distribution is
            determined for `galname`. These are different for each galaxy but 
            are based on `maxv0` and `Nvs`, which are
            the same for every galaxy.
        f[galname]['ps']: The probability densities at the given
            speeds. 
        f['v_v0']: The v/v0's used, which are the same for every 
            galaxy
        f['params']: The parmeter values used in determining v0.
            These are the medians of the chains from emcee.

    Parameters
    ----------
    df_source: str
        The file name of the analysis-results DataFrame to use. The circular
        speed to use in determining v0_gal comes from here.
    mcmc_samples_source: str
        The name of the emcee results / samples file.
    maxv0: float, default 2.3
        The highest multiple of v0_gal to evaluate.
    Nvs: int, default=30
        The number of speeds to evaluate between 0 and v0 * maxv0
    tgt_fname_override: str, default None
        If specified, the name of the file into which to save the distribution
        samples. If the user doesn't specify a file name, the method will save
        the distribution samples in 
        'distrib_samples_' + today + '(' + mcmc_samples_source + ').h5'

    Returns
    -------
    None
    '''
    import dm_den
    import h5py
    import paths
    import datetime
    import numpy as np
    from progressbar import ProgressBar

    flat_samples = read(mcmc_samples_source)
    #params = estimate(mcmc_samples_source)
    params = {t: np.percentile(flat_samples[:, i], 50.) 
            for i, t in enumerate(['d', 'e', 'h', 'j', 'k'])}
    print(params)
    indices = np.random.randint(len(flat_samples), size=5000)
    THETA = flat_samples[indices]

    nondisks = ['m12z', 'm12w']
    df = dm_den.load_data(df_source).drop(nondisks)

    v_by_v0_bins = np.linspace(0., maxv0, Nvs + 1)
    vs_by_v0 = (v_by_v0_bins[1:] + v_by_v0_bins[:-1]) / 2.

    today = datetime.datetime.today().strftime('%d%m%Y')
    tgt_fname = (
        'distrib_samples_by_v0_' 
        + today 
        + '(' 
        + mcmc_samples_source.replace('.h5', '').replace('mcmc_samples_', '')
        + ').h5'
    )
    with h5py.File(paths.data + tgt_fname, 'w') as f:
        f.create_dataset('v_v0', data=vs_by_v0)
        params_grp = f.create_group('params')
        for p in params:
            params_grp.create_dataset(p, data=params[p])

        pbar = ProgressBar()
        for galname in pbar(df.index):
            gal = f.create_group(galname)
            vc = df.loc[galname, 'v_dot_phihat_disc(T<=1e4)']
            v0 = params['d'] * (vc / 100.) ** params['e']
            vs = vs_by_v0 * v0
            ps = make_gal_distrib_samples(
                    df, 
                    galname,
                    THETA,
                    vs
            )
            gal.create_dataset('vs', data=vs)
            gal.create_dataset('ps', data=ps)
    return None
