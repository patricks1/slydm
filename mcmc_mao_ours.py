theta_ranges = {
    'd': [10., 130.],
    'e': [0.8, 5.],
    'p': [1., 5.]
}

def calc_log_likelihood(theta, X, ys, dys):
    '''
    Parameters
    ----------
    theta: np.ndarray, shape = (3,)
        Model parameters [d, e, p]
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
        speed v, where f(v) is our sigmoid-damped speed distribution.
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
    d, e, p = theta
    vcs, vescs, vs = X.T
    v0s = d * (vcs / 100.) ** e
    yhats = fitting.mao(vs, v0s, vescs, p)

    chi2 = np.sum((yhats - ys) ** 2. / dys ** 2.) # chi squared

    if np.isnan(chi2):
        # Some non-allowed parameter values might result in a f(v) = 0/0.
        # These aren't allowed anyway, so just return -np.inf.
        return -np.inf

    log_likelihood = -chi2
    return log_likelihood

def run(df_source,
        tgt_fname,
        pdfs_source='v_pdfs_disc_dz1.0_20240606.pkl',
        vesc_fit_source='data_raw.pkl'):
    import dm_den
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

    #return calc_log_likelihood(np.array([100., 1., 0.5]), X, ys, dys)

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

        sampler.run_mcmc(pos, int(7e4), progress=True)
