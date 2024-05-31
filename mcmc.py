def calc_log_likelihood(theta, X, ys):
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
    '''
    import numpy as np
    import fitting

    Ndimy = ys.ndim
    if Ndimy != 1:
        raise Exception('ys should only have one dimension.')
    N = len(ys)
    if X.shape != (N, 2):
        raise Exception('X has the wrong shape. It should have 2 columns and'
                        ' the same number'
                        ' of rows as y.')
    d, e, h, j, k = theta 
    vcs, vs = X.T
    v0s = d * (vcs / 100.) ** e 
    vdamps = h * (vcs / 100.) ** j
    yhats = [fitting.smooth_step_max(v, v0, vdamp, k, speedy=True)
             for v, v0, vdamp in zip(vs, v0s, vdamps)]
    yhats = np.array(yhats)
    sse = np.sum((yhats - ys) ** 2.)

    if np.isnan(sse):
        # Some non-allowed parameter values might result in a f(v) = 0/0 e.g. 
        # negative
        # k values. These aren't allowed anyway, so just return -np.inf.
        return -np.inf

    log_likelihood = -sse
    return log_likelihood

def calc_log_gaussian_prior(theta):
    '''
    Calculate the log prior assuming a p(theta) is a multivariate Gaussian.
    '''
    import scipy
    import paths
    import pickle
    import numpy as np

    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        ls_result = pickle.load(f)

        # Best estimate of the parameters from least squares minimization
        mu = np.array([ls_result[param] 
                       for param in ['d', 'e', 'h', 'j', 'k']])

        cov = ls_result['covar'] * 5. 
    return scipy.stats.multivariate_normal.logpdf(theta, mean=mu, cov=cov)

def calc_log_uniform_prior(theta):
    '''
    Calculate the log prior assuming a uniform distribution.

    Parameters
    ----------
    theta: np.ndarray, shape=(5,)
        Parameter values for which to return the probability density. They
        should be ordered as d, e, h, j, k.
    '''
    import numpy as np

    theta_ranges = {
            'd': [50., 500.],
            'e': [0.1, 5.],
            'h': [50., 500.],
            'j': [0.1, 5.],
            'k': [1.e-5, 0.5]
    }

    N = 0. # Normalization
    for t in theta_ranges:
        N += theta_ranges[t][1] - theta_ranges[t][0]
    p = 1. / N
    
    # Ensure that the prior integrates over all dimensions to unity.
    P = 0.
    for t in theta_ranges:
        P += (theta_ranges[t][1] - theta_ranges[t][0]) * p
    assert P == 1.

    for i, t in enumerate(theta_ranges):
        # Check if each parameter is within its acceptable range. If not,
        # the prior is 0, and the log prior is -infty.
        in_range = theta_ranges[t][0] <= theta[i] <= theta_ranges[t][1]
        if not in_range:
            return -np.inf

    return np.log(p)

def calc_log_post(theta, X, ys, log_prior_function):
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
    log_prior_function: function
        The log prior function to use. For example, 
        `mcmc.calc_log_gaussian_prior`
        or `mcmc.calc_log_uniform_prior`.
    '''
    import numpy as np

    log_prior_value = log_prior_function(theta)
    log_likelihood_value = calc_log_likelihood(theta, X, ys)
    log_posterior_value = log_prior_value + log_likelihood_value

    return log_posterior_value

def run(log_prior_function, df_source, tgt_fname):
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

    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        ls_result = pickle.load(f)

        # Best estimate of the parameters from least squares minimization
        mu = np.array([ls_result[param] 
                       for param in ['d', 'e', 'h', 'j', 'k']])

        # Purposefully starting off in the wrong place to see what happens
        #mu = np.array([90., 1., 250., 1., 0.01])
         
    nondisks = ['m12z', 'm12w']
    with open(paths.data + 'v_pdfs_disc_dz1.0.pkl', 'rb') as f:
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

    # Make a tiny Gaussian ball around the current best estimate (I don't know
    # why we name this `pos`, but that's what 
    # https://github.com/dfm/emcee/blob/main/docs/tutorials/line.ipynb calls
    # it.)
    ndim = len(mu) # number of parameters
    nwalkers = 32
    pos = mu + 1.e-4 * np.random.randn(nwalkers, ndim)

    with multiprocessing.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, calc_log_post,
                                        args=(X, ys, log_prior_function),
                                        backend=backend,
                                        pool=pool)
        #print('initial size: {0}'.format(backend.iteration))
        #print('starting samples shape: {0}'
        #      .format(sampler.get_chain().shape))

        sampler.run_mcmc(pos, int(5e3), progress=True)

    return None
