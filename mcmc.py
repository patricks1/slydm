
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
    sse = (yhats - ys) ** 2.
    log_likelihood = -sse
    return log_likelihood

def calc_log_prior(theta):
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

        cov = ls_result['covar']
    return scipy.stats.multivariate_normal.logpdf(theta, mean=mu, cov=cov)

def calc_log_post(theta, X, ys):
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
    '''
    log_prior = calc_log_prior(theta)
    if not np.isfinite(log_prior):
        return -np.inf
    return log_prior + calc_log_likelihood(theta, X, ys)

if __name__ == '__main__':
    import emcee
    import pickle
    import paths
    import dm_den
    import numpy as np
    
    backend = emcee.backends.HDFBackend(paths + 'mcmc_samples.h5')

    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        ls_result = pickle.load(f)

        # Best estimate of the parameters from least squares minimization
        mu = np.array([ls_result[param] 
                       for param in ['d', 'e', 'h', 'j', 'k']])
    with open(paths.data + 'v_pdfs_disc_dz1.0.pkl', 'rb') as f:
        pdfs = pickle.load(f)
    df = dm_den.load_data('dm_stats_dz1.0_20231211.h5')

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


    sampler = emcee.EnsembleSampler(nwalkers, ndim, calc_log_post,
                                    args=(X, ys),
                                    backend=backend)
    sampler.run_mcmc(pos, int(5e3), progress=True)
