theta_ranges = {
    'p': [1., 5.]
}

def calc_log_likelihood(theta, X, ys, dys):
    '''
    Parameters
    ----------
    theta: np.ndarray, shape = (1,)
        Model parameters [p]
    X: np.ndarray, shape = (N, 3)
        Feature matrix where N is the number of data rows, the 0 column is 
        circular speed of a galaxy,
        the 1 column is the escape speed estimates v_esc(v_c), and the 2 
        column
        is particle speeds in the speed distribution.
    ys: np.ndarray, shape = (N,)
        Target vector 4*pi*v^2 * f(v) probability density of a DM particle 
        having 
        a given
        speed v, where f(v) is the Mao speed distribution.
    dys: np.ndarray, shape = (N,)
        Errors in y
    '''
    import fitting
    import numpy as np

    Ndimy = ys.ndim
    if Ndimy != 1 or dys.ndim != 1:
        raise Exception('ys and dys should only have one dimension.')
    N = len(ys)
    if len(dys) != N:
        raise Exception('ys and dys should be be the same length.')
    if X.shape != (N, 3):
        raise Exception('X has the wrong shape. It should have 3 columns and'
                        ' the same number'
                        ' of rows as y.')
    if len(theta) != 1:
        raise Exception('`theta` should only have one component: p.')

    p = theta[0]
    vcs, vescs, vs = X.T
    yhats = fitting.mao(vs, vcs, vescs, p)

    chi2 = np.sum((yhats - ys) ** 2. / dys ** 2.) # chi squared

    if np.isnan(chi2):
        # Some non-allowed parameter values might result in a f(v) = 0/0.
        # These aren't allowed anyway, so just return -np.inf.
        return -np.inf

    log_likelihood = -chi2 / 2.
    return log_likelihood

def run(df_source,
        tgt_fname,
        pdfs_source='v_pdfs_disc_dz1.0_20240606.pkl',
        vesc_fit_source='data_raw.pkl',
        nsteps=int(1.3e5)):
    from . import dm_den
    import mcmc
    import os
    import emcee
    import pickle
    import paths
    import multiprocessing
    import numpy as np

    # Turn off numpy's multiprocessing that can cause problems
    os.environ['OMP_NUM_THREADS'] = '1'
    
    backend = emcee.backends.HDFBackend(paths.data + tgt_fname) 

    nondisks = ['m12z', 'm12w']
    with open(paths.data + pdfs_source, 'rb') as f:
        pdfs = pickle.load(f)
    for nondisk in nondisks:
        del pdfs[nondisk]
    df = dm_den.load_data(df_source).drop(nondisks)

    # Feature matrix
    with open(paths.data + vesc_fit_source, 'rb') as f:
        vesc_fit = pickle.load(f)

    for galname in pdfs:
        dict_gal = pdfs[galname]
        vc_gal = df.loc[galname, 'v_dot_phihat_disc(T<=1e4)']
        vesc_vc_gal = (
            vesc_fit['vesc_amp'] * (vc_gal / 100.) ** vesc_fit['vesc_slope']
        )
        dict_gal['vcirc'] = np.repeat(vc_gal, len(dict_gal['ps']))
        dict_gal['vesc_vc'] = np.repeat(vesc_vc_gal, len(dict_gal['ps']))
    vs_truth = np.concatenate([pdfs[galname]['vs']
                   for galname in pdfs])
    vcircs = np.concatenate([
        pdfs[galname]['vcirc']
        for galname in pdfs
    ])
    vescs_vc = np.concatenate([
        pdfs[galname]['vesc_vc']
        for galname in pdfs
    ])

    X = np.array([vcircs, vescs_vc, vs_truth]).T

    # Target vector of true probabilities of dm particles having the speeds 
    # `vs_truth`
    ys = np.concatenate([pdfs[galname]['ps']
                   for galname in pdfs])
    dys = np.concatenate([pdfs[galname]['p_errors']
                         for galname in pdfs])

    # With Poisson errors, a y_i == 0 causes dy_i == np.nan
    isfinite = np.isfinite(dys)
    X = X[isfinite]
    ys = ys[isfinite]
    dys = dys[isfinite]

    #return calc_log_likelihood(np.array([101., 1., 0.5]), X, ys, dys)

    nwalkers = 64 
    ndim = len(theta_ranges) # number of parameters

    log_prior_function = mcmc.calc_log_uniform_prior
    log_prior_args = (theta_ranges,)
    with multiprocessing.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, mcmc.calc_log_post,
                                        args=(
                                            X, 
                                            ys, 
                                            dys,
                                            log_prior_function,
                                            log_prior_args, 
                                            calc_log_likelihood
                                        ),
                                        backend=backend,
                                        pool=pool)

        size = backend.iteration
        print('initial size: {0}'.format(size))

        if size > 0:
            # Start the walkers where they last ended.
            pos = None
        else:
            # Generate initial positions within the valid ranges
            pos = np.zeros((nwalkers, ndim))
            for i, key in enumerate(theta_ranges):
                    pos[:, i] = np.random.uniform(
                            theta_ranges[key][0], 
                            theta_ranges[key][1], 
                            nwalkers
                    )

        sampler.run_mcmc(pos, nsteps, progress=True)

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
    update_paper: bool, default False
        Whether to update the LaTeX paper in paths.paper

    Returns
    -------
    results_dict: dict
        The best estimates of the parameters from the chains.
    '''
    from . import dm_den
    from . import dm_den_viz
    import staudt_utils
    import read_mcmc
    import UCI_tools.tools as uci
    import numpy as np
    from IPython.display import display, Math

    param_keys = ['p']

    samples = read_mcmc.read(samples_fname)

    ndim = samples.shape[1]
    results_dict = {}
    for i in range(ndim):
        est = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(est)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.4f}}}^{{+{2:.4f}}}"
        txt = txt.format(est[1], q[0], q[1], param_keys[i])
        display(Math(txt))
        results_dict[param_keys[i]] = est[1]
        results_dict['d' + param_keys[i]] = np.array([q[0], q[1]])
    if result_fname is not None:
        dm_den.save_var_raw(results_dict, result_fname) 
    if update_paper:
        for key in param_keys:
            # Save strings to be used in paper.tex
            y = results_dict[key]
            DY = results_dict['d' + key]
            # y_txt is a string. DY_TXT is an array of strings, or just an
            # array of just one string. Either way, 
            # type(DY_TXT) == np.ndarray, which is why we're denoting it in
            # all caps.
            y_txt, DY_TXT = staudt_utils.sig_figs(y, DY)
            uci.save_prediction(key + '_mao_naive_agg', y_txt,  DY_TXT)
    return results_dict
