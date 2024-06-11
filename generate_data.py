import datetime
import mcmc
import read_mcmc
import paths
import fitting
import dm_den

date_str = datetime.today().strftime('%Y%m%d')
vescphi_dict_fname = date_str + '_vescs(phi)_rot' + '.pkl'
df_fname = date_str + '_dm_stats_dz1.0' '.h5'
pdfs_fname = date_str + '_v_pdfs_disc_dz1.0' + '.pkl'
ls_results_fname = date_str + '_data_raw' + '.pkl' # least-squares results
mcmc_samples_fname = date_str + '_mcmc_samples' + '.h5'
mcmc_distrib_samples_fname = date_str + '_mcmc_distrib_samples' + '.h5'

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
# Generate least squares fit results for the Staudt et al. speed distribution.
###############################################################################
_ = fitting.plt_universal(
        gals='discs', 
        err_method=None,
        update_values=True,
        pdfs_fname=pdfs_fname,
        raw_results_fname=ls_results_fname,
)
_ = fitting.save_samples(df_fname, N=5000)

###############################################################################
# MCMC
###############################################################################
# Run the MCMC. This will take >12 hr.
mcmc.run(
        mcmc.calc_log_gaussian_prior,
        df_fname, 
        mcmc_samples_fname,
        pdfs_fname,
        ls_results_source=ls_results_fname,
        log_prior_args=(ls_results_fname,) 
)
# Sample the `THETA` posteior to make speed distribution samples.
read_mcmc.make_distrib_samples(
        df_fname,
        mcmc_distrib_samples_fname
)
