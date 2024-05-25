import datetime
import mcmc
import read_mcmc
import paths
import fitting

date_str = datetime.today().strftime('%Y%m%d')
df_source = 'dm_stats_dz1.0_20231211.h5'

###############################################################################
# Generate least squares fit results for the Staudt et al. speed distribution.
###############################################################################
_ = fitting.plt_universal(
        gals='discs', 
        err_method=None,
        update_values=True
)
_ = fitting.save_samples(df_source, N=5000)

###############################################################################
# MCMC
###############################################################################
mcmc_samples_fname = 'mcmc_samples_' + date_str + '.h5'
mcmc_distrib_samples_fname = 'mcmc_distrib_samples_' + date_str + '.h5'

# Run the MCMC. This will take >12 hr.
mcmc.run(
        df_source, 
        mcmc_samples_fname
)
# Sample the THETA posteior to make speed distribution samples.
read_mcmc.make_distrib_samples(
        df_source,
        mcmc_distrib_samples_fname
)
