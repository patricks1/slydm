def read(samples_fname, consider_burnin=True, flat=True):
    import emcee
    import paths
    import warnings
    import numpy as np

    reader = emcee.backends.HDFBackend(paths.data + samples_fname)
    if consider_burnin:
        try:
            tau = reader.get_autocorr_time()
            thin = int(0.5 * np.min(tau))
        except emcee.autocorr.AutocorrError as err:
            tau = reader.get_autocorr_time(quiet=True)
            thin = 1
        burnin = int(2 * np.max(tau))
        samples = reader.get_chain(discard=burnin, thin=thin, flat=flat)
        print('burn-in: {0:0.0f}'
              '\nthin: {1:0.0f}'
              .format(burnin, thin))
    else:
        samples = reader.get_chain(flat=flat)

    print('\nsamples shape: {0}'
          .format(str(samples.shape)))

    return samples

def corner_plot(samples_fname, consider_burnin=True, log_prior_function=None):
    import corner
    import scipy
    import copy
    import matplotlib.pyplot as plt
    import numpy as np
    from progressbar import ProgressBar

    flat_samples = read(samples_fname, consider_burnin=consider_burnin)
    ndim = flat_samples.shape[1]
    fig = corner.corner(flat_samples, labels=['D', 'e', 'H', 'j', 'k'])
    axs = np.array(fig.axes).reshape((ndim, ndim))

    if log_prior_function is not None:
        # For each plot on the diagonal, marginalize over all the other
        # parameters.

        # Initialize a placeholder for the ranges that the corner plot 
        # currently shows for each of i
        # the 
        # parameters
        xranges = np.zeros((ndim, 2))

        # Initialize a placeholder for all parameter values that the corner 
        # plot currently shows for each 
        # of the 
        # parameters
        Xs = []

        for i in range(ndim):
            # For each parameter, get the values that the corner plot
            # currently shows.
            ax = axs[i, i]
            patches = ax.patches
            polygon = patches[0]
            vertices = polygon.get_path().vertices
            xs = vertices[:, 0] # All theta[i] values
            Xs.append(xs)
            
            # Min and max theta[i] values
            xranges[i] = np.array([xs.min(), xs.max()]) 

        for i in range(1):
            def func(n0, n1, n2, n3, ti):
                '''
                Convert the log_prior_function so the nuicance parameters are
                the first four args and the parameter of interest is last so
                scipy.integrate.nquad can integrate over the nuicance
                parameters
                '''
                theta = np.array([n0, n1, n2, n3])
                theta = np.insert(theta, i, ti)
                return log_prior_function(theta)


            ax = axs[i, i]
            xs = Xs[i]
            logys = np.zeros(len(xs))

            # Our integration bounds correspond only to the nuicance parameters
            # i.e. not the i'th parameter
            nuicance_ranges = np.delete(xranges, i, axis=0)

            # Perform the marginalization.
            pbar = ProgressBar()
            for j, x in enumerate(pbar(xs)):
                logys[j] = scipy.integrate.nquad(
                        func, nuicance_ranges, args=(x,)
                )[0]
            ys = np.exp(np.array(logys))

            ax_twin = ax.twinx()
            ax_twin.plot(xs, ys)

    plt.show()

    return None

def estimate(samples_fname):
    from IPython.display import display, Math
    import numpy as np

    samples = read(samples_fname)
    labels = ['D', 'e', 'H', 'j', 'k']

    ndim = samples.shape[1]
    for i in range(ndim):
        est = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(est)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
        txt = txt.format(est[1], q[0], q[1], labels[i])
        display(Math(txt))
    return None

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
