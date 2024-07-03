import datetime
import mcmc
import read_mcmc
import paths
import fitting
import dm_den

date_str = datetime.today().strftime('%Y%m%d')
vescphi_dict_fname = date_str + '_vescs(phi)_rot.pkl'
df_fname = date_str + '_dm_stats_dz1.0.h5'
pdfs_fname = date_str + '_v_pdfs_disc_dz1.0.pkl'
pdfs4systematics_fname = date_str + '_v_by_v0_pdfs_disc_dz1.0.pkl'
ls_results_fname = date_str + '_ls_data_raw.pkl' # least-squares results
mao_naive_aggp_results_fname = date_str + '_mao_naive_aggp_data_raw.pkl'
mcmc_samples_fname = date_str + '_mcmc_samples.h5'
mcmc_distrib_samples_fname = date_str + '_mcmc_distrib_samples.h5'
mcmc_distrib_samples_by_v0_fname = date_str + '_mcmc_distrib_samples_by_v0.h5'
mcmc_results_fname = date_str + '_mcmc_results.pkl'

###############################################################################
# Generate galaxy properties dataframe
###############################################################################
_ = dm_den.get_v_escs(vescphi_dict_fname, rotate=True)
_ = dm_den.gen_data(
        df_fname, 
        dr=1.5, 
        dz=1., 
        source='cropped', 
        vescphi_dict_fname=vescphi_dict_fname
)

###############################################################################
# Generate probability densities p(v) = 4 pi v^2 f(v) where f(v) is the speed
# distribution
###############################################################################
_ = dm_den.make_v_pdfs(r=8.3, dr=1.5, dz=1., fname=pdfs_fname)

###############################################################################
# Generate least squares fit results
###############################################################################
# Staudt
_ = fitting.plt_universal(
    gals='discs', 
    err_method=None,
    update_values=False,
    pdfs_fname=pdfs_fname,
    raw_results_fname=ls_results_fname,
)
#_ = fitting.save_samples(df_fname, N=5000)

# Mao
_ = fitting.fit_mao_naive_aggp(
    vcut_type='lim_fit', 
    df_source=df_fname, 
    raw_results_fname=mao_naive_aggp_results_fname
)
_ = fitting.fit_mao(
    vcut_type='lim_fit',
    df_source=df_fname,
    update_values=True
)

###############################################################################
# MCMC
###############################################################################
# Run the MCMC.
mcmc.run(
    mcmc.calc_log_gaussian_prior,
    df_fname, 
    mcmc_samples_fname,
    pdfs_fname,
    ls_results_source=ls_results_fname,
    log_prior_args=(ls_results_fname,) 
)
# Generate best parameter estimates from the MCMC
read_mcmc.estimate(mcmc_samples_fname, mcmc_results_fname, update_paper=True)
# Sample the `THETA` posteior to make speed distribution samples.
read_mcmc.make_distrib_samples(
        df_fname,
        mcmc_distrib_samples_fname
)
# Also generate speed distribution *samples* for a universal array of v/v0 
# values
read_mcmc.make_distrib_samples_by_v0(
        df_fname,
        mcmc_samples_fname,
        mcmc_distrib_samples_by_v0_fname,
        maxv0=2.3,
        Nvs=30
)
# And make the *actual distributions* for that v/v0 array
dm_den.make_v_over_v0_pdfs(
        df_source,
        mcmc_results_fname,
        fname=pdfs4systematics_fname,
        maxv0=2.3,
        Nbins=30,
        r=8.3,
        dr=1.5,
        dz=1.)

###############################################################################
# Calculate RMS's and save them to the paper
###############################################################################
_ = fitting.calc_rms_all_methods(
    df_fname,
    pdfs_fname,
    staudt_results_fname=mcmc_results_fname, 
    mao_naive_aggp_results_fname=mao_naive_aggp_results_fname,
    update_paper=True
)

