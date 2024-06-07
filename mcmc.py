theta_ranges = {
        'd': [100., 130.],
        'e': [0.8, 1.2],
        'h': [200., 400.],
        'j': [0.1, 0.8],
        'k': [1.e-2, 4.5e-2]
}

wide_ranges = {
        'd': [0., 500.],
        'e': [0.1, 3.],
        'h': [50., 1000.],
        'j': [0.1, 3.],
        'k': [0.1e-2, 8.e-2]
}

def calc_log_likelihood(theta, X, ys, dys):
    '''
    Parameters
    ----------
    theta: np.ndarray, shape = (5,)
        Model parameters [d, e, h, j, k]
    X: np.ndarray, shape = (N, 2)
        Feature matrix where N is the number of data rows, the 0 column is 
        circular speed of a galaxy, and 
        the 1 column
        is particle speeds in the speed distribution.
    ys: np.ndarray, shape = (N,)
        Target vector 4*pi*v^2 * f(v) probability density of a DM particle 
        having 
        a given
        speed v, where f(v) is our sigmoid-damped speed distribution.
    dys: np.ndarray, shape = (N,)
        Errors in y
    '''
    import numpy as np
    import fitting

    Ndimy = ys.ndim
    if Ndimy != 1 or dys.ndim != 1:
        raise Exception('ys and dys should only have one dimension.')
    N = len(ys)
    if len(dys) != N:
        raise Exception('ys and dys should be be the same length.')
    if X.shape != (N, 2):
        raise Exception('X has the wrong shape. It should have 2 columns and'
                        ' the same number'
                        ' of rows as y.')
    d, e, h, j, k = theta 
    vcs, vs = X.T
    v0s = d * (vcs / 100.) ** e 
    vdamps = h * (vcs / 100.) ** j
    yhats = fitting.smooth_step_max(vs, v0s, vdamps, k, speedy=True)
    
    chi2 = np.sum((yhats - ys) ** 2. / dys ** 2.) # chi squared

    if np.isnan(chi2):
        # Some non-allowed parameter values might result in a f(v) = 0/0 e.g. 
        # negative
        # k values. These aren't allowed anyway, so just return -np.inf.
        return -np.inf

    log_likelihood = -chi2
    return log_likelihood

def calc_log_gaussian_prior(theta, multiplier, ls_results_source):
    '''
    Calculate the log prior assuming a p(theta) is a multivariate Gaussian.
    '''
    import scipy
    import paths
    import pickle
    import numpy as np

    ndim = len(theta)

    with open(paths.data + ls_results_source, 'rb') as f:
        ls_result = pickle.load(f)

        # Best estimate of the parameters from least squares minimization
        mu = np.array([ls_result[param] 
                       for param in ['d', 'e', 'h', 'j', 'k']])

        cov = np.zeros((ndim, ndim))
        dis = np.diag_indices(ndim)
        np.fill_diagonal(cov, ls_result['covar'][dis])
        cov = cov * multiplier 
    return scipy.stats.multivariate_normal.logpdf(theta, mean=mu, cov=cov)

def calc_log_fat_gaussian_prior(theta, ls_results_source):
    return calc_log_gaussian_prior(theta, 50.**2., ls_results_source)

def calc_log_wide_gaussian_prior(theta, ls_results_source):
    return calc_log_gaussian_prior(theta, 5.**2., ls_results_source)

def calc_log_wide_uniform_prior(theta):
    return calc_log_uniform_prior(theta, wide_ranges)

def calc_log_narrower_uniform_prior(theta):
    return calc_log_uniform_prior(theta, theta_ranges)

def calc_log_uniform_prior(theta, theta_ranges):
    '''
    Calculate the log prior assuming a uniform distribution.

    Parameters
    ----------
    theta: np.ndarray, shape=(5,)
        Parameter values for which to return the probability density. They
        should be ordered as d, e, h, j, k.
    '''
    import numpy as np

    N = 0. # Normalization
    for t in theta_ranges:
        N += theta_ranges[t][1] - theta_ranges[t][0]
    p = 1. / N
    
    # Ensure that the prior integrates over all dimensions to unity.
    P = 0.
    for t in theta_ranges:
        P += (theta_ranges[t][1] - theta_ranges[t][0]) * p
    assert np.allclose(P, 1.)

    for i, t in enumerate(theta_ranges):
        # Check if each parameter is within its acceptable range. If not,
        # the prior is 0, and the log prior is -infty.
        in_range = theta_ranges[t][0] <= theta[i] <= theta_ranges[t][1]
        if not in_range:
            return -np.inf

    return np.log(p)

def calc_log_post(theta, X, ys, dys, log_prior_function, log_prior_args=()):
    '''
    Calculate the log posterior

    Parameters
    ----------
    theta: np.ndarray, shape = (5,)
        Possible model parameters [d, e, h, j, k]
    X: np.ndarray, shape = (N, 2)
        Feature matrix where N is the number of data rows, the 0 column is 
        circular speed of a galaxy, and 
        the 1 column
        is particle speeds in the speed distribution.
    ys: np.ndarray, shape = (N,)
        Target vector 4*pi*v^2 * f(v) probability density of a DM particle 
        having 
        a given
        speed v, where f(v) is our sigmoid-damped speed distribution.
    dys: np.ndarray, shape = (N,)
        Errors in y
    log_prior_function: function
        The log prior function to use. For example, 
        `mcmc.calc_log_gaussian_prior`
        or `mcmc.calc_log_uniform_prior`.
    log_prior_args: tuple, default ()
        Args that should be passed to the log prior function, other than the
        `theta` vector
    '''
    import numpy as np

    log_prior_value = log_prior_function(theta, *log_prior_args)
    log_likelihood_value = calc_log_likelihood(theta, X, ys, dys)
    log_posterior_value = log_prior_value + log_likelihood_value

    return log_posterior_value

def run(log_prior_function, df_source, tgt_fname, 
        pdfs_source='v_pdfs_disc_dz1.0_20240606.pkl',
        ls_results_source='data_raw.pkl',
        log_prior_args=()):
    import emcee
    import pickle
    import paths
    import dm_den
    import argparse
    import os
    import multiprocessing
    import numpy as np

    # Turn off numpy's multiprocessing that can cause problems
    os.environ['OMP_NUM_THREADS'] = '1'
    
    backend = emcee.backends.HDFBackend(paths.data + tgt_fname) 

    with open(paths.data + ls_results_source, 'rb') as f:
        ls_result = pickle.load(f)

        # Best estimate of the parameters from least squares minimization
        mu = np.array([ls_result[param] 
                       for param in ['d', 'e', 'h', 'j', 'k']])

        # Purposefully starting off in the wrong place to see what happens
        #mu = np.array([90., 1., 250., 1., 0.01])
         
    nondisks = ['m12z', 'm12w']
    with open(paths.data + pdfs_source, 'rb') as f:
        pdfs = pickle.load(f)
    for nondisk in nondisks:
        del pdfs[nondisk]
    df = dm_den.load_data(df_source).drop(nondisks)

    # Feature matrix
    for galname in pdfs:
        dict_gal = pdfs[galname]
        vc_gal = df.loc[galname, 'v_dot_phihat_disc(T<=1e4)']
        dict_gal['vcirc'] = np.repeat(vc_gal, len(dict_gal['ps']))
    vs_truth = np.concatenate([pdfs[galname]['vs']
                   for galname in pdfs])
    vcircs = np.concatenate([pdfs[galname]['vcirc']
                         for galname in pdfs])
    X = np.array([vcircs, vs_truth]).T

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

    nwalkers = 64 
    ndim = len(mu) # number of parameters

    with multiprocessing.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, calc_log_post,
                                        args=(
                                            X, ys, dys, log_prior_function,
                                            log_prior_args),
                                        backend=backend,
                                        pool=pool)

        size = backend.iteration
        print('initial size: {0}'.format(size))
        #print('starting samples shape: {0}'
        #      .format(sampler.get_chain().shape))

        if size > 0:
            # Start the walkers where they last ended.
            pos = None
        else:
            # Set the initial positions to a tiny Gaussian ball around the 
            # best estimate from least-squares minimization.
            pos = mu + 1.e-4 * np.random.randn(nwalkers, ndim)

            # Generate initial positions within the valid ranges
            #pos = np.zeros((nwalkers, ndim))
            #for i, key in enumerate(theta_ranges):
            #        pos[:, i] = np.random.uniform(
            #                theta_ranges[key][0], 
            #                theta_ranges[key][1], 
            #                nwalkers
            #        )

        sampler.run_mcmc(pos, int(7e4), progress=True)

    return None
