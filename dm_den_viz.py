from adjustText import adjust_text
from IPython.display import display, Latex
import numpy as np
import pandas as pd
import cmasher as cmr
import dm_den
import sys
import paths
import staudt_utils
import pickle
import itertools
import math
import grid_eval
import grid_eval_mao
import lmfit
import copy
import os
import UCI_tools.tools as uci
from UCI_tools import staudt_tools
from progressbar import ProgressBar

import scipy
from scipy.stats.stats import pearsonr

from astropy import units as u
from astropy import constants as c
from astropy.units import cds
cds.enable()

import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['font.family'] = 'serif' 
#rcParams['xtick.labelsize'] = 16
#rcParams['ytick.labelsize'] = 16
#rcParams['axes.grid']=True
rcParams['axes.titlesize']=24
#rcParams['axes.labelsize']=20
rcParams['axes.titlepad']=15
rcParams['legend.frameon'] = True
rcParams['legend.facecolor']='white'
rcParams['figure.facecolor'] = (1., 1., 1., 1.) #white with alpha=1.

max_naive_color = 'C0'
max_fit_color = '#a903fc' 
mao_prediction_color = 'c'
mao_naive_color = '#8d8d8d'

def plotter_old(gals, dat, gal_names, datloc, ylabel,
            yscale='linear', adjustment=None, figsize=(7,8)):
    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    texts=[] #make a container for labels that adjust_text will tune
    for d,g,name in zip(dat, gals, gal_names):
        y=d[datloc]
        if adjustment=='log':
            y=np.log10(y)
        elif adjustment is None:
            pass
        else:
            raise ValueError
        x=np.log10(d[0])
        ax.plot(x,y,'bo',label='m12'+g)
        #add annotation to texts
        texts+=[ax.annotate(name,(x*1.001,y*1.0001), fontsize=14)]
    ax.set_xlabel(log_rho_solar_label)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.grid(True,which='both')
    
    if adjustment=='log':
        corr=pearsonr(np.log10(dat[:,datloc]),dat[:,0])
    else:
        corr=pearsonr(dat[:,datloc],dat[:,0])
    ax.annotate('corr$\,={0:0.2f}$'.format(corr[0]),
                (0,-0.2),
                xycoords='axes fraction',
                fontsize=15)
    adjust_text(texts)
    #ax.legend(bbox_to_anchor=(1.04,0.5), loc='center left')
    plt.show()

disp_vir_label = '$\sigma(R_\mathrm{vir})' \
                 '\ [\,\mathrm{km}\cdot\mathrm{s}^{-1}\,]$'
disp_solar_label='$\sigma(R_0)\ [\,\mathrm{km}\cdot\mathrm{s}^{-1}\,]$'
disp_dm_solar_label = '$\sigma_\mathrm{DM}(R_0)' \
                      '\ [\,\mathrm{km}\cdot\mathrm{s}^{-1}\,]$'
disp_gas_solar_label = '$\sigma_\mathrm{gas}(R_0)' \
                       '\ [\,\mathrm{km}\cdot\mathrm{s}^{-1}\,]$'
log_disp_solar_label = '$\log(\,\sigma(R_0)' \
                       '\,/\,[\,\mathrm{km}\cdot\mathrm{s}^{-1}\,]\,)$'
m_label='$M_\mathrm{vir}\ [\mathrm{M}_\odot]$'
mbtw_label='$M(10\,\mathrm{kpc}<r<R_\mathrm{vir})\[\mathrm{M}_\odot]$'
log_rho_solar_label='$\log(\,\\rho(R_0)\,/\,[\,\mathrm{M}_\odot'\
                    '\mathrm{kpc}^{-3}\,]\,)$'
log_rho_label = '$\log(\,\\rho(R_\mathrm{vir})' \
                '\,/\,[\,\mathrm{M}_\odot\mathrm{kpc}^{-3}\,]\,)$'
rho_label='$\\rho(R_\mathrm{vir})\;[\,\mathrm{M}_\odot\mathrm{kpc}^{-3}\,]$'
den_label = '$\\rho\,/\,\\left[\mathrm{M_\odot kpc^{-3}}\\right]$'
disp_label = '$\\sigma_\mathrm{3D}\,/\,'\
             '\\left[\mathrm{km\,s^{-1}}\\right]$'
gmr_label = '$\sqrt{GM/R_0}\,/\,'\
              '\\left[\mathrm{km\,s^{-1}}\\right]$'
vc_label = '$v_\mathrm{c}\,/\,[\mathrm{km\,s^{-1}}]$'

vcut_labels = {'lim_fit': '$v_\mathrm{esc}(v_\mathrm{c})$',
               'lim': '$v_\mathrm{esc}$',
               'vesc_fit': ('$v_{\\rm esc}(\Phi'
                            '\\rightarrow v_\mathrm{c})$'),
               'veschatphi': '$v_\mathrm{esc}(\Phi)$',
               'ideal': '$v_\mathrm{cut, ideal}$'}

# Y-axis limit for all residual plots
resids_lim = 0.9

# vc from Eilers et al. 2019
vc_eilers = 229.
dvc_eilers = 7.

vesc_mw = 550.

# Recommendations from Baxter et al. 2021
vesc_std = 544. # km/s
vc_std = 238. # km/s
v0_std = vc_std

# vc ranges from Sofue 2020
vc_sofue=238.
dvc_sofue=14.
log_dvc_neg = np.log10(vc_sofue/(vc_sofue-dvc_sofue))
log_dvc_pos = np.log10((vc_sofue+dvc_sofue)/vc_sofue)

# Density ranges from Sofue 2020
rho_sofue = 0.39*u.GeV/c.c**2.*u.cm**-3.
drho_sofue = 0.09*u.GeV/c.c**2.*u.cm**-3.
rho_sofue = rho_sofue.to(u.M_sun*u.kpc**-3.).value
drho_sofue = drho_sofue.to(u.M_sun*u.kpc**-3.).value
rho_min_sofue = np.log10(rho_sofue-drho_sofue)
rho_max_sofue = np.log10(rho_sofue+drho_sofue)

def plt_slr(fname, xcol, ycol,
            xlabel,ylabel,
            xadjustment=None, yadjustment=None,
            xscale='linear', yscale='linear', 
            figsize=(7,6), dpi=100,
            showlabels=True,
            labelsize=15, arrowprops=None, formula_y=-0.2,
            dropgals=None, show_formula=True, adjust_text_kwargs={},
            tgt_fname=None, ax_slr_kwargs={}):
    'Plot a simple linear regression'

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    res = ax_slr(ax, fname, xcol, ycol,
                 xlabel,ylabel,
                 xadjustment, yadjustment,
                 xscale, yscale, 
                 showlabels,
                 labelsize, arrowprops, formula_y,
                 dropgals, show_formula=show_formula, 
                 adjust_text_kwargs=adjust_text_kwargs, 
                 **ax_slr_kwargs)

    if tgt_fname is not None:
        plt.draw()
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=140)
    plt.show()

    if 'x_forecast' in ax_slr_kwargs:
        y_hat = res[-1]
        #print(y_hat)
        return y_hat
    else:
        return None

def plt_forecast(ax, X_forecast, Yhat, dYhat, xadjustment):
    '''
    Parmeters
    ---------
    ax: matplotlib.axes
        The axes on which to plot the forecast.
    X_forecast: np.ndarray, shape=(number of datapoints forecasted,
                                   number of features)
        Feature matrix, i.e. independent variable(s) vector, with which the
        forecast was made.
    Y_hat: np.ndarray, shape=(number of datapoints forecasted, 1)
        Forecast vector. dm_den.mlr keeps forecast vectors in 2D so it can do
        matrix math with them, so we write this function to take those 2D
        vectors directly. When not inputting dm_den.mlr outputs directly into
        this function, it may be necessary for the user to massage their 
        forecast
        float into a 2D vector.
    dYhat: np.array
        Error bars for Y_hat.
        - shape(number of datapoints forecasted, 1): Symmetric.
        - shape(number of datapoints forecasted, 2): Asymmetric. Column 0 is
          dYhat_plus. Column 1 is dYhat_minus
    xadjustment: {None, 'log'}
        Specifies whether the plot is showing log values or unadjusted values.
        In either case, the user should provide an unadjusted feature matrix
        X_forecast. The function will make the adjustment if specified.

    Returns
    ------
    eb: list
        A list of errorbar objects
    '''
    if xadjustment=='log':
        X_forecast = np.log10(X_forecast)
    else:
        X_forecast = np.array(X_forecast)
    N = len(X_forecast)
    eb = []
    colors = [plt.cm.cool(i) 
              for i in np.linspace(0, 
                                   1, 
                                   N)]
    colors = itertools.cycle(['r']+colors)
    for i in range(N):
        color = next(colors)
        if len(dYhat[i]) == 2:
            # asymmetric error
            # plt.errorbar's yerr argument is supposed to be (yerr_minus,
            # yerr_plus), which is the opposite order of our dYhat.
            yerr = np.array([[dYhat[i][1]], [dYhat[i][0]]])
        elif len(dYhat[i]) == 1:
            # symmetric error
            yerr = dYhat[i][0]
        else:
            raise ValueError('dYhat[{0:0.0f}] has an unexpected shape.'
                             .format(i))
        #err_fr_observations = staudt_utils.linear2log(
        #        10. ** Yhat[i, 0], 
        #        0.055 * 10. ** Yhat[i, 0])[1:]
        #
        #yerr = np.sqrt(yerr**2. + err_fr_observations**2.)
        #if len(yerr) > 1:
        #    yerr = yerr.reshape(2, -1)
        eb_add = ax.errorbar(X_forecast.flatten()[i], 
                             Yhat[i,0],
                             yerr=yerr,
                             c=color, capsize=3,
                             marker='o', ms=8, 
                             mec=color, mfc=color)
        eb += [eb_add[0]]

    for x, y, error in zip(X_forecast, Yhat, dYhat):
        if not isinstance(error, (list, np.ndarray)):
            display(Latex('$y(x={0:0.4f})={1:0.4f}\pm {2:0.4f}$'\
                          .format(x[0], y[0], error[0])))
        else:
            if len(error) == 1:
                display(Latex('$y(x={0:0.4f})={1:0.4f}\pm {2:0.4f}$'\
                              .format(x[0], y[0], error[0])))
            elif len(error) == 2:
                display(Latex('$y(x={0:0.4f})={1:0.4f}'
                              '^{{+{2:0.4f}}}'
                              '_{{-{3:0.4f}}}$'\
                              .format(x[0], y[0], error[0], error[1])))
            else:
                raise ValueError('Error array has the wrong number of '
                                 'elements.')

    return eb 

def make_formula_appear(show_formula,
                        xcol, ycol,
                        intercept, coefs, r2,
                        reg_xscale, reg_yscale,
                        xadjustment, yadjustment, ax=None, formula_y=-0.4):
    if show_formula == True and ax is None:
        raise ValueError('If show_formula is True, the user must provide an '
                         'ax.')
    formula_strings = {'vcirc_R0':'v_\mathrm{c}',
                       'v_cool_gas':'v_\\phi',
                       'disp_dm_solar':'\sigma_\mathrm{{DM}}', 
                       'den_solar':'\\rho_\mathrm{{DM}}',
                       'v_dot_phihat_disc(T<=1e4)': 'v_\mathrm{c}',
                       'disp_dm_disc_cyl': '\sigma',
                       'den_disc': '\\rho'
                       }
    
    def get_strs():
        try:
            xstring = formula_strings[xcol]
        except:
            xstring = 'x'
        try:
            ystring = formula_strings[ycol]
        except:
            ystring = 'y'
        return xstring, ystring

    if reg_xscale == 'log' and reg_yscale=='log':
        # if plotting log data on both axes, show the formula of the form 
        # y=Ax^m
        
        if intercept <= 1.:
            amplitude_str = staudt_utils.mprint(10.**intercept,
                                                d=1,
                                                show=False).replace('$','')
        
        else:
            amplitude_str = '10^{{{0:0.1f}}}'.format(intercept)
        
        xstring, ystring = get_strs()
        if show_formula=='outside':
            display(Latex('${3:s}={0:s}\,{4:s}^{{{1:0.2f}}}$'
                        .format(amplitude_str, 
                                coefs[0], r2, ystring, xstring)))
            display(Latex('$r^2_\mathrm{{log\,space}}={2:0.2f}$'\
                        .format(amplitude_str, 
                                coefs[0], r2, ystring, xstring)))
        else:
            ax.annotate('${3:s}={0:s}\,{4:s}^{{{1:0.2f}}}$\n'
                        '$r^2_\mathrm{{log\,space}}={2:0.2f}$'\
                        .format(amplitude_str, 
                                coefs[0], r2, ystring, xstring),
                        (0,formula_y),
                        xycoords='axes fraction', fontsize=18)
    elif xadjustment is None and yadjustment is None:
        xstring, ystring = get_strs()
        if intercept<0.:
            operator = '-'
        else:
            operator = '+'
        if show_formula=='outside':
            display(Latex('${0:s}={1:0.2f}{2:s}{5:s}{3:0.2f}$'
                        .format(ystring, coefs[0], xstring, 
                                np.abs(intercept), r2,
                                operator)))
            display(Latex('$r^2={4:0.2f}$'
                        .format(ystring, coefs[0], xstring, 
                                np.abs(intercept), r2,
                                operator)))
        else:
            ax.annotate('${0:s}={1:0.4f}{2:s}{5:s}{3:0.2f}$\n'
                        '$r^2={4:0.2f}$'\
                        .format(ystring, coefs[0], xstring, 
                                np.abs(intercept), r2,
                                operator),
                        (0., formula_y),
                        xycoords='axes fraction', fontsize=18)
    return None

def ax_slr(ax, fname, xcol, ycol,
           xlabel,ylabel,
           xadjustment=None, yadjustment=None,
           xscale='linear', yscale='linear', 
           showlabels=True,
           labelsize=15, arrowprops=None, formula_y=-0.2,
           dropgals=None, showGeV=True, show_formula=True,
           x_forecast=None, dX=None, forecast_sig=1.-0.682, verbose=False,
           adjust_text_kwargs={}, legend_txt=None, 
           return_error=False, show_band=False, linear_dyfrac_data=0.,
           **kwargs):
    '''
    Plot a simple linear regression on ax

    Noteworthy parameters
    ---------------------
    linear_dyfrac_data: float, default 0.
        The fractional uncertainty, in linear units, in the observed values of 
        the target vector. No matter on what scale we perform the regression,
        this percentage error applies to the uncertainty in the target vector
        in *linear* space.
    '''

    #Perform the regression in linear space unless we're plotting log data, as
    #opposed to plotting unadjusted data but on a log scale
    if xadjustment == 'log':
        reg_xscale = 'log'
        display_xadj = 'log'
    elif xadjustment == 'logreg_linaxunits':
        # If we want the regression to be log but then display the x-axis in
        # linear units (note that we may still want to scale the displayed 
        # x-axis in log-scale, which would be set by the `xscale` variable)
        reg_xscale = 'log'
        display_xadj = None
    elif xadjustment is None:
        reg_xscale = 'linear'
        display_xadj = None
    else:
        raise ValueError('x adjustment should be \'log\''
                         ', \'logreg_linaxunits\' or None')
    if yadjustment == 'log':
        reg_yscale = 'log'
        display_yadj = 'log'
    elif yadjustment == 'logreg_linaxunits':
        # If we want the regression to be log but then display the y-axis in
        # linear units (note that we may still want to scale the displayed 
        # y-axis in log-scale, which would be set by the `yscale` variable)
        reg_yscale = 'log'
        display_yadj = None
    elif yadjustment is None:
        reg_yscale = 'linear'
        display_yadj = None
    else:
        raise ValueError('y adjustment should be \'log\', '
                         '\'logreg_linaxunits\' or None')

    mlr_res = dm_den.mlr(fname, xcols=[xcol], 
                         ycol=ycol,
                         xscales=[reg_xscale], yscale=reg_yscale,
                         dropgals=dropgals, 
                         prediction_x=x_forecast, fore_sig=forecast_sig, dX=dX,
                         verbose=verbose, return_band=True, 
                         return_coef_errors=True, **kwargs)
    # delta_beta is the fit parameter errors
    coefs, intercept, r2, Xs, ys, ys_fit, r2a, resids, delta_beta, band \
            = mlr_res[:10]

    if show_band:
        band_disp = band.copy()
        if xadjustment == 'logreg_linaxunits':
            band_disp[0] = 10.**band_disp[0]
        if yadjustment == 'logreg_linaxunits':
            band_disp[1:] = 10.**band_disp[1:]

        # band[-1] is just the Y_hat @ the X used for the band, so we don't
        # use it in ax.fill_between.
        ax.fill_between(*band_disp[:-1], color='grey', alpha=0.2, lw=0.)

    if x_forecast is not None:
        prediction_y = mlr_res[-1] #[Y, Y uncertainty] 
        if yadjustment == 'logreg_linaxunits':
            prediction_y = np.array(prediction_y, dtype=object)
            Ys = np.zeros(prediction_y.shape[1:])
            dYs = np.zeros((prediction_y[1].shape[0], 2))
            for i, (Y, dY) in enumerate(zip(prediction_y[0], 
                                            prediction_y[1])):
                conversion = staudt_utils.log2linear(Y[0], dY[0],
                                                     check_symmetry=False)
                Ys[i,0] = conversion[0]

                varY_fr_Ydata = (conversion[0] * linear_dyfrac_data) ** 2.

                # We set check_symmetry=False, so 
                # dYs[i] = np.array([dy_plus, 
                #                    dy_minus])
                dYs[i] = np.sqrt(conversion[1:] ** 2. +  varY_fr_Ydata)
        elif yadjustment == 'log':
            Ys = prediction_y[0]
            dYs = np.zeros((prediction_y[1].shape[0], 2))

            linear_Y = 10. ** prediction_y[0]
            # linear-space uncertainty in the target vector predictions sourced
            # from uncertainty in the target vector observations on which the
            # model was built:
            linear_dY_fr_data = linear_dyfrac_data * linear_Y

            for i, (logy, dlogy) in enumerate(zip(prediction_y[0], 
                                                  prediction_y[1])):
                # logy is shaped (1,)
                # dlogy is shaped (1,) (because before doing any 
                #     transformations, the error is symmetric)
                conversion = staudt_utils.log2linear(
                        logy[0], dlogy[0])

                max_linear_y = 10. ** (logy[0] + dlogy[0])
                min_linear_y = 10. ** (logy[0] - dlogy[0])
                linear_y = 10. ** logy
                linear_dy_fr_forecast = np.array([max_linear_y - linear_y[0],
                                                  linear_y[0] - min_linear_y])

                linear_dY = np.sqrt(linear_dy_fr_forecast ** 2. 
                                    + linear_dY_fr_data ** 2.)

                max_linear_y = linear_y[0] + linear_dY[i, 0]
                min_linear_y = linear_y[0] - linear_dY[i, 1]
                max_log_y = np.log10(max_linear_y)
                min_log_y = np.log10(min_linear_y)

                dlogy = np.array([max_log_y - logy[0], logy[0] - min_log_y])
                dYs[i] = dlogy

                # linear_dY is shaped as (1, 1 if symmetric
                #                            2 if asymmetric)
                # so we take linear_dY[i] to get at the error values
                before = staudt_utils.sig_figs(
                    linear_y[0], linear_dY[i])
                exponentiated_dlogy = np.array(
                        [10. ** max_log_y - linear_y[0],
                         linear_y[0] - 10. ** min_log_y])
                after = staudt_utils.sig_figs(
                    linear_y[0], exponentiated_dlogy)
                for elements in zip(before, after):
                    if not np.array_equal(*elements):
                        raise ValueError('The exponentiated (linear-space)'
                                         ' prediction {0} is significantly'
                                         ' different from the'
                                         ' exponentiation of its log {1}.'
                                         ' This'
                                         ' will be a problem when we'
                                         ' exponentiate the log prediction'
                                         ' to cite the prediction in'
                                         ' linear space.'.format(before,
                                                                 after))
        else:
            # If everything, including the regression, is linear:
            Ys = prediction_y[0]
            linear_dY_fr_data = linear_dyfrac_data * Ys
            dYs = np.sqrt(prediction_y[1] ** 2. + linear_dY_fr_data ** 2.)
        eb = plt_forecast(ax, x_forecast, Ys, dYs, xadjustment)
        adjust_text_kwargs['objects'] = eb

    df = dm_den.load_data(fname)
    if dropgals:
        df = df.drop(dropgals)

    ###########################################################################
    # Fill plot with galaxy points
    ###########################################################################
    if xadjustment=='log':
        xlabel=loglabel(xlabel)
    if display_yadj=='log':
        ylabel=loglabel(ylabel)
    fill_ax_new(ax, df, xcol, ycol, 
                xlabel=xlabel,
                ylabel=ylabel, 
                xscale=xscale,
                yscale=yscale,
                xadjustment=display_xadj,
                yadjustment=display_yadj,
                showcorr=False,
                arrowprops=arrowprops, showlabels=showlabels, 
                adjust_text_kwargs=adjust_text_kwargs,
                labelsize=labelsize, **kwargs)
    ###########################################################################
    
    if yadjustment == 'logreg_linaxunits':
        ys_fit = 10.**ys_fit
    if xadjustment == 'logreg_linaxunits':
        Xs[0] = 10.**Xs[0]
    ax.plot(Xs[0], ys_fit, label=legend_txt) #Plot the regression line

    if show_formula:
        make_formula_appear(show_formula,
                            xcol, ycol,
                            intercept, coefs, r2,
                            reg_xscale, reg_yscale,
                            xadjustment, yadjustment, ax=ax, 
                            formula_y=formula_y)

    den_cols = ['den_solar','den_disc','den_shell'] 

    if xcol in den_cols and showGeV:
        # If density is on the x axis, put particle units on the top of the
        # plot
        showGeV_x(ax, xadjustment)

    if ycol in den_cols and showGeV: 
        # If density is on the y axis, put particle units on the right of the
        # plot
        showGeV_y(ax, display_yadj)

    result = [coefs, intercept, resids]
    if return_error:
        pos = band[1] - band[3]
        neg = band[3] - band[2]
        pos_neg = np.concatenate((pos,neg)).flatten()
        # Add the fit coefficient error array (delta_beta) and the average
        # half width 
        # of the
        # error band (pos_neg.mean()) to the return. The width of the error
        # band varies at different x values, but this just returns the avg
        # of the half widths. Presenting the simple average like this could 
        # be problematic if the width varies significantly and if someone might
        # want to evaluate yhat(x) at an x far from the average and quote the
        # error we present.
        result += [delta_beta, pos_neg.mean()]
    if x_forecast is not None:
        result += [[Ys, dYs]]
    return tuple(result)

def showGeV_x(ax, xadjustment):
    ax2=ax.twiny()
    x0, x1 = ax.get_xlim()
    visible_ticks = np.array([t for t in ax.get_xticks() \
                              if t>=x0 and t<=x1])
    if xadjustment=='log':
        msun_kpc = 10.**visible_ticks*u.M_sun/u.kpc**3.
    elif xadjustment is None:
        msun_kpc = visible_ticks*u.M_sun/u.kpc**3.
    else:
        raise ValueError('Adjustment should be \'log\' or None')
    labs = msun_kpc.to(u.GeV/cds.c**2./u.cm**3.)
    ax2.set_xlim(x0,x1)
    ax2.set_xticks(visible_ticks)
    ax2.set_xticklabels(['{:,.2f}'.format(lab.value) for lab in labs], 
                        fontsize=12.)
    ax2.set_xlabel('$\\rho\,/\,\left[\mathrm{GeV}\,\mathrm{cm^{-3}}\\right]$',
                   fontsize=12.)
    ax2.grid(False)
    return None

def showGeV_y(ax, yadjustment):
    ax2=ax.twinx()
    x0, x1 = ax.get_ylim()
    lim = np.array([x0, x1]) 

    if yadjustment=='log':
        lim_msun_kpc = 10.**lim * u.M_sun/u.kpc**3.
    elif yadjustment is None:
        lim_msun_kpc = lim * u.M_sun/u.kpc**3.
    else:
        raise ValueError('Adjustment should be \'log\' or None')
    lim_GeV = lim_msun_kpc.to(u.GeV/cds.c**2./u.cm**3.)
    labs = np.array([t for t in np.arange(lim_GeV[0].value - 0.1, 
                                          lim_GeV[1].value + 0.1,
                                          0.04) \
                         if t>=lim_GeV[0].value and t<=lim_GeV[1].value])

    labs *= u.GeV/cds.c**2./u.cm**3.
    visible_ticks = labs.to(u.M_sun/u.kpc**3.)
    if yadjustment == 'log':
        visible_ticks = np.log10(visible_ticks.value)
    else:
        visible_ticks = visible_ticks.value

    ax2.set_ylim(x0,x1)
    ax2.set_yticks(visible_ticks)
    ax2.set_yticklabels(['{:,.2f}'.format(lab.value) for lab in labs], 
                        fontsize=12.)
    ax2.set_ylabel('$\\rho\,/\,\left[\mathrm{GeV}\,\mathrm{cm^{-3}}\\right]$', 
                   fontsize=12.)
    ax2.grid(False)
    return None

def showGeV_y_old(ax, yadjustment):
    ax2=ax.twinx()
    x0, x1 = ax.get_ylim()
    visible_ticks = np.array([t for t in ax.get_yticks() \
                              if t>=x0 and t<=x1])
    if yadjustment=='log':
        msun_kpc = 10.**visible_ticks*u.M_sun/u.kpc**3.
    elif yadjustment is None:
        msun_kpc = visible_ticks*u.M_sun/u.kpc**3.
    else:
        raise ValueError('Adjustment should be \'log\' or None')
    labs = msun_kpc.to(u.GeV/cds.c**2./u.cm**3.)
    ax2.set_ylim(x0,x1)
    ax2.set_yticks(visible_ticks)
    ax2.set_yticklabels(['{:,.2f}'.format(lab.value) for lab in labs], 
                        fontsize=12.)
    ax2.set_ylabel('$\mathrm{GeV}\,c^{-2}\,\mathrm{cm^{-3}}$', 
                   fontsize=12.)
    ax2.grid(False)
    return None

def plotter(df, ycol, ylabel,
            yscale='linear', yadjustment=None, figsize=(5,6), showlabels=True,
            labelsize=15, arrowprops=None, xcol='den_solar', 
            xlabel=None, xadjustment='log'):
    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    if isinstance(df,(list,tuple)):
        for d in df:
            fill_ax(ax,d,ycol,ylabel,yscale,yadjustment,showlabels,labelsize,
                    arrowprops, xcol=xcol, xlabel=xlabel, 
                    xadjustment=xadjustment)
    else:
        fill_ax(ax,df,ycol,ylabel,yscale,yadjustment,showlabels,labelsize,
                arrowprops, xcol=xcol, xlabel=xlabel, 
                xadjustment=xadjustment)
    fig.patch.set_facecolor('white')
    plt.show()
    return None

def fill_ax_new(ax, df, xcol, ycol, 
                xlabel, 
                ylabel,
                xadjustment=None, yadjustment=None,
                xscale='linear', yscale='linear', 
                showlabels=True,
                labelsize=15, arrowprops=None, color='blue', alpha=1., 
                showcorr=True, legend_txt=None, 
                adjust_text_kwargs={}, xtickspace=None, ytickspace=None,
                **kwargs):
    if xcol == 'den_solar' and xadjustment == 'log' and xlabel is None:
        xlabel = log_rho_solar_label
    xs=df[xcol]
    if xadjustment=='log':
        xs=np.log10(xs)
    elif xadjustment is not None:
        raise ValueError
    ys=df[ycol]
    if yadjustment=='log':
        ys=np.log10(ys)
    elif yadjustment is None:
        pass
    else:
        raise ValueError

    if color=='masses':
        c = dm_den.load_data('dm_stats_stellar_20221110.h5') \
                  .loc[df.index,['mvir_stellar']].values
        #######################################################################
        #import cropper
        #for galname in df.index:
        #    gal_data = cropper.load_data(galname, getparts=['PartType4'])
        #    M0 = dm_den.get_mwithin(8.3, 
        #                            gal_data['PartType4']['r'],
        #                            gal_data['PartType4']['mass_phys'])
        #    df.loc[galname, 'M0_stellar'] = M0
        #
        #c = df['M0_stellar'].values
        #######################################################################
        c = np.log10(c)
        cmap = cmr.get_sub_cmap('viridis', 0., 0.9)
        sc = ax.scatter(xs,ys,marker='o',c=c,alpha=alpha,label=legend_txt,
                        cmap=cmap)
        if ycol == 'den_disc':
            pad = 0.15
        else:
            pad = 0.05
        cb = plt.colorbar(sc, pad=pad, 
                          location='right')
        cb.ax.tick_params(labelsize=12)
        cb.set_label(size=12,
                     label='$\log M_\mathrm{\star}\,/\,\mathrm{M_\odot}$')
        #######################################################################
        #cb.set_label(
        #     size=12,
        #     label='$\log M(r<R_0)_\mathrm{\star}\,/\,\mathrm{M_\odot}$')
        #######################################################################
    else:
        ax.plot(xs,ys,'o',color=color,alpha=alpha,label=legend_txt)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(False,which='both')
    
    if showcorr:
        corr=pearsonr(xs,ys)
        ax.annotate('corr$\,={0:0.2f}$'.format(corr[0]),
                    (0,-0.25),
                    xycoords='axes fraction',
                    fontsize=15)
    if showlabels:
        texts=[] #make a container for labels that adjust_text will tune
        for x,y,name in zip(xs, ys, df.index):
            #add annotation to texts
            texts+=[ax.annotate(name, (float(x), float(y)), 
                                fontsize=labelsize)]
        if arrowprops:
            adjust_text(texts, ax=ax, 
                        **adjust_text_kwargs, arrowprops=arrowprops)
        else:
            adjust_text(texts, ax=ax, **adjust_text_kwargs)

    if yscale == 'log':
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
    if xscale == 'log':
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
    if xtickspace is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=xtickspace))
        ax.xaxis.set_minor_locator(plt.NullLocator())
    if ytickspace is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=ytickspace))
        ax.yaxis.set_minor_locator(plt.NullLocator())

    return None

def fill_ax(ax,df, ycol, ylabel,
            yscale='linear', yadjustment=None, showlabels=True,
            labelsize=15, arrowprops=None, color='blue', alpha=1., 
            showcorr=True, label=None, xcol='den_solar', 
            xlabel=None, xadjustment='log'):
    if xcol == 'den_solar' and xadjustment == 'log' and xlabel is None:
        xlabel = log_rho_solar_label
    xs=df[xcol]
    if xadjustment=='log':
        xs=np.log10(xs)
    ys=df[ycol]
    if yadjustment=='log':
        ys=np.log10(ys)
    elif yadjustment is None:
        pass
    else:
        raise ValueError
    ax.plot(xs,ys,'o',color=color,alpha=alpha,label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.grid(True,which='both')
    
    if showcorr:
        corr=pearsonr(xs,ys)
        ax.annotate('corr$\,={0:0.2f}$'.format(corr[0]),
                    (0,-0.25),
                    xycoords='axes fraction',
                    fontsize=15)
    if showlabels:
        texts=[] #make a container for labels that adjust_text will tune
        for x,y,name in zip(xs, ys, df.index):
            #add annotation to texts
            texts+=[ax.annotate(name, (float(x*1.001), float(y*1.0001)), 
                                fontsize=labelsize)]
        if arrowprops:
            adjust_text(texts, arrowprops=arrowprops, ax=ax)
        else:
            adjust_text(texts, ax=ax)
    return None

def make_plot_feed(df):
    feed=[{'df':df,
           'col':'disp_vir',
           'ylabel':disp_vir_label,
           'yscale':'linear'},
          {'df':df,
           'col':'disp_solar',
           'ylabel':disp_solar_label,
           'yscale':'linear'},
          {'df':df,
           'col':'mvir_calc',
           'ylabel':m_label,
           'yscale':'log'},
          {'df':df,
           'col':'m10tovir',
           'ylabel':mbtw_label,
           'yscale':'log'},
          {'df':df,
           'col':'den_vir',
           'ylabel':log_rho_label,
           'yscale':'linear',
           'adjustment':'log'},
          {'df':df,
           'col':'den_vir',
           'ylabel':rho_label,
           'yscale':'linear',
           'adjustment':None}]
    return feed

def comp_disp_vc(galname='m12i', dr=1.5, rundate='20210730'):
    fname='disp_vc_{0:s}_{1:s}_dr001.5.h5'.format(galname,rundate)
    rs, disps_fire, vcs_fire, disps_dmo, vcs_dmo = utils.load_disp_vc(fname)
    fig=plt.figure(figsize=(7,5))
    ax=fig.add_subplot(111)
    ax.plot(rs, disps_fire, color='#1f77b4', 
            label='$\sigma_\mathrm{3D, FIRE}$')
    ax.plot(rs, vcs_fire, 'b-', label='$v_\mathrm{circ, FIRE}$')
    ax.plot(rs, disps_dmo, 'k--', label='$\sigma_\mathrm{3D, DMO}$')
    ax.plot(rs, vcs_dmo, color='grey', linestyle='--',
            label='$v_\mathrm{circ, DMO}$')
    ax.set_title(galname)
    ax.set_xscale('log')
    ax.set_ylabel('$\mathrm{km}\cdot\mathrm{s}^{-1}$')
    ax.set_xlabel('$r\:[\mathrm{kpc}]$')
    ax.legend(fontsize=16.)
    fig.patch.set_facecolor('white')
    plt.show()
    return None

def loglabel(label):
    'wrap "log(...)" around given label'

    label = label.replace('$','')
    label = '$\log\\left(\,'+label+'\\right)$'
    return label

def make_err_bars_fr_resids(ax, reg):
    '''
    Make rror bars from residuals
    '''
    resids = reg[2]
    delta_neg = np.percentile(resids, (1.-0.682)/2.*100.)
    delta_pos = np.percentile(resids, (1.-(1.-0.682)/2.)*100.)
    delta = np.mean(np.abs((delta_neg, delta_pos)))
    
    ax.errorbar(np.log10(vc_sofue), 
                reg[-1][0], #The last element of reg will be the prediction.
                yerr=delta, 
                marker='o', ms=8, c='k', mec='r', mfc='r', capsize=3)
    
    return None

def draw_yshade(ax, ycol):
    if ycol=='den_disc':
        bounds = (rho_min_sofue, rho_max_sofue)
        ax.axhspan(*bounds, 
                    alpha=0.2, color='gray', ls='none')
    return None

def draw_xshade(ax, vc, dvc, x_shade_mult=1., **kwargs):
    xlo = np.log10(vc-dvc)
    xhi = np.log10(vc+dvc)
    if x_shade_mult != 1.:
        # Increase width of shaded band.
        x = np.log10(vc)
        xlo = x - x_shade_mult*(x-xlo)
        xhi = x + x_shade_mult*(xhi-x)
    ax.axvspan(xlo, 
               xhi, 
               alpha=0.4, color='green', ls='none')
    return None

def draw_shades(ax, ycol, vc, dvc, xmult=1.):
    draw_yshade(ax, ycol)
    draw_xshade(ax, vc, dvc, xmult)
    return None

def plt_vesc_vc_vs_vc(dfsource, figsize=(4.5, 4.8), labelsize=11, 
                   adjust_text_kwargs={}, formula_y=-0.3, dpi_show=120,
                   xtickspace=None, ytickspace=None, label_overrides={},
                   marker_label_size=11,
                   show_formula=True,
                   update_values=False, tgt_fname=None, verbose=False,
                   show_vesc=False):
    import dm_den
    df = dm_den.load_data(dfsource)

    vcut_d = dm_den.find_last_v()
    df_vcut = pd.DataFrame.from_dict(vcut_d, orient='index', columns=['vcut'])
    df = pd.concat([df, df_vcut], axis=1)
    
    df_source = 'dm_stats_w_vcut.h5'
    dm_den.save_data(df, df_source)
    
    ycol = 'vcut'
    xcol = 'vc100'
    df.drop(['m12z', 'm12w'], inplace=True)
    fig = plt.figure(figsize=figsize, dpi=dpi_show)
    ax = fig.add_subplot(111)
    X_forecast = np.array([[vc_eilers / 100.]]) # MW vcirc
    xadjustment='logreg_linaxunits'
    yadjustment='logreg_linaxunits'
    P1sigma = scipy.special.erf(1. / np.sqrt(2)) # ~68%

    reg = dm_den.mlr(df_source, xcols=[xcol], ycol=ycol,
                     xscales=['log'], yscale='log',
                     dropgals=['m12z', 'm12w'], 
                     prediction_x=X_forecast,
                     dX=np.array([[dvc_eilers / 100.]]),
                     fore_sig=1.-P1sigma,
                     beta_sig=1.-P1sigma,
                     return_band=True,
                     return_coef_errors=True,
                     verbose=verbose)

    # delta_beta is the fit parameter errors
    coefs, log_intercept, r2, Xs, ys, ys_pred, r2a, resids, delta_beta, band \
            = reg[:10]
    slope = coefs[0]

    # reg[-1] has two components; reg[-1][0] is the forecast; reg[-1][1] is the
    # uncertainty on the forecast. In order to be able to do the matrix math
    # in multiple regression, ax_slr keeps reg[-1][0] as 2 dimensional, but we
    # can just take the [0,0] element (0th datapoint, 0th column). 
    vesc_hat_mw = reg[-1][0][0,0] # Predicted MW vesc

    # mlr proves dyhat as 2 dimensional in
    # case the uncertainty on y is asymmetric and in case we're forecasting
    # multiple datapoints. We can take the 0 element of reg[-1][1] (the 0th 
    # datapoint) but want to keep all columuns
    # of reg[-1][1][0] in case the uncertainty is asymmetric.
    dvesc_mw = reg[-1][1][0]

 
    # Errors on the fit parameters (i.e. on the beta vector)
    dbeta = reg[-3] 

    ax.fill_between(10. ** band[0] * 100., *(10. ** band[1:]), 
                    color='grey', alpha=0.2,
                    lw=0.)
    # Plot the best fit line
    ax.plot(10. ** band[0] * 100., 
            10. ** band[-1], '-')
    fill_ax_new(ax, df, 
                'v_dot_phihat_disc(T<=1e4)', ycol, 
                xlabel=vc_label,
                ylabel=('${0:s}\,/\,\left[\mathrm{{km\,s^{{-1}}}}\\right]$'
                        .format(vcut_labels['lim'].replace('$', '')
                                                  .replace('{','{{')
                                                  .replace('}','}}'))),
                xscale='log', yscale='log',
                color='masses',
                labelsize=marker_label_size,
                xtickspace=xtickspace, 
                ytickspace=ytickspace,
                showcorr=False, adjust_text_kwargs=adjust_text_kwargs,
                arrowprops={'arrowstyle': '-'})
    vesc_hat_mw_transform = staudt_utils.log2linear(vesc_hat_mw, dvesc_mw[0]) 
    plt_forecast(ax, X_forecast * 100., 
                 Yhat=np.array([[vesc_hat_mw_transform[0]]]),
                 dYhat=np.array([vesc_hat_mw_transform[1:]]),
                 xadjustment=None)
    ax.annotate('Milky Way', (vc_eilers, vesc_hat_mw_transform[0]), 
                fontsize=labelsize,
                color='red', 
                arrowprops={'arrowstyle':'-|>', 'color':'red'}, 
                textcoords='axes fraction',
                xytext=(0.45, 0.7))

    # Replace the necessary data labels/annotations with their overrides
    override_labels(label_overrides, ax, dm_den.load_data(df_source),
                    ycol, 'v_dot_phihat_disc(T<=1e4)', labelsize )

    if show_vesc:
        ax.plot(df['v_dot_phihat_disc(T<=1e4)'], df['vesc'], 'bo', ms=3,
        label=vcut_labels['veschatphi'])
        df_vcut = pd.DataFrame.from_dict(vcut_d, orient='index', 
                                         columns=['vlim'])
        df = pd.concat([df, df_vcut], axis=1)

        for gal in df.index:
            ax.annotate('',
                        xy=(df.loc[gal, 'v_dot_phihat_disc(T<=1e4)'],
                        df.loc[gal, 'vlim'] + 2.), 
                        xycoords='data',
                        xytext =(df.loc[gal, 'v_dot_phihat_disc(T<=1e4)'],
                         df.loc[gal, 'vesc']),
                        textcoords = 'data',
                        arrowprops={'arrowstyle': '-|>', 'ls': 'dashed', 
                        'color':'blue', 'alpha':0.4}, size=10.)

    if show_formula:
        make_formula_appear(show_formula,
                            xcol, ycol,
                            log_intercept, coefs, r2,
                            reg_xscale='log', reg_yscale='log',
                            xadjustment=xadjustment, yadjustment=yadjustment)

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=350)
    plt.show()

    if update_values:
        # Save the MW vesc prediction in the LaTeX data
        vesc_hat_mw_txt, dvesc_mw_txt = staudt_utils.sig_figs(
                vesc_hat_mw_transform[0], vesc_hat_mw_transform[1:])
        uci.save_prediction('vesc_mw(vc)', vesc_hat_mw_txt, dvesc_mw_txt)

        # Save the amplitude to the LaTeX data
        dlog_intercept = dbeta[0][0]
        intercept_transform = staudt_utils.log2linear(log_intercept, 
                                                      dbeta[0][0])
        amp = intercept_transform[0]
        damp = intercept_transform[1:]
        amp_str, damp_str = staudt_utils.sig_figs(amp, damp)
        uci.save_prediction('veschat_amp', amp_str, damp_str)

        # Save the slope to the LaTeX data
        slope_str, dslope_str = staudt_utils.sig_figs(slope, dbeta[1][0])
        uci.save_prediction('veschat_slope', slope_str, dslope_str)

        # Save the vesc(vc) predictions
        df = dm_den.load_data(df_source)
        if xadjustment in ['logreg_linaxunits', 'log'] \
           and yadjustment in ['logreg_linaxunits', 'log']:
            vesc_hat_dict = dict(amp * df[xcol] ** slope)
        vesc_hat_dict['mw'] = vesc_hat_mw_transform[0]
        with open(paths.data + 'vcut_hat_dict.pkl', 'wb') as f:
            pickle.dump(vesc_hat_dict, f, pickle.HIGHEST_PROTOCOL)

    os.remove(paths.data + df_source)

    return None

def plt_vesc_pot_vs_vc(df_source, figsize=(4.5, 4.8), labelsize=11, 
                   adjust_text_kwargs={}, formula_y=-0.3, dpi_show=120,
                   xtickspace=None, ytickspace=None, label_overrides={},
                   marker_label_size=11,
                   update_values=False, tgt_fname=None, verbose=False):
    '''
    \hat{vesc}(Phi) vs vc
    '''
    ycol = 'vesc'
    xcol = 'vc100'
    df = dm_den.load_data(df_source)
    df.drop(['m12z', 'm12w'], inplace=True)
    fig = plt.figure(figsize=figsize, dpi=dpi_show)
    ax = fig.add_subplot(111)
    X_forecast = np.array([[vc_eilers / 100.]]) # MW vcirc
    xadjustment='logreg_linaxunits'
    yadjustment='logreg_linaxunits'
    P1sigma = scipy.special.erf(1. / np.sqrt(2)) # ~68%

    reg = dm_den.mlr(df_source, xcols=[xcol], ycol=ycol,
                     xscales=['log'], yscale='log',
                     dropgals=['m12z', 'm12w'], 
                     prediction_x=X_forecast,
                     dX=np.array([[dvc_eilers / 100.]]),
                     fore_sig=1.-P1sigma,
                     beta_sig=1.-P1sigma,
                     return_band=True,
                     return_coef_errors=True,
                     verbose=verbose)

    # delta_beta is the fit parameter errors
    coefs, log_intercept, r2, Xs, ys, ys_pred, r2a, resids, delta_beta, band \
            = reg[:10]
    slope = coefs[0]

    # reg[-1] has two components; reg[-1][0] is the forecast; reg[-1][1] is the
    # uncertainty on the forecast. In order to be able to do the matrix math
    # in multiple regression, ax_slr keeps reg[-1][0] as 2 dimensional, but we
    # can just take the [0,0] element (0th datapoint, 0th column). 
    vesc_hat_mw = reg[-1][0][0,0] # Predicted MW vesc

    # mlr proves dyhat as 2 dimensional in
    # case the uncertainty on y is asymmetric and in case we're forecasting
    # multiple datapoints. We can take the 0 element of reg[-1][1] (the 0th 
    # datapoint) but want to keep all columuns
    # of reg[-1][1][0] in case the uncertainty is asymmetric.
    dvesc_mw = reg[-1][1][0]

 
    # Errors on the fit parameters (i.e. on the beta vector)
    dbeta = reg[-3] 

    ax.fill_between(10. ** band[0] * 100., *(10. ** band[1:]), 
                    color='grey', alpha=0.2,
                    lw=0.)
    # Plot the best fit line
    ax.plot(10. ** band[0] * 100., 
            10. ** band[-1], '-')
    fill_ax_new(ax, df, 
                'v_dot_phihat_disc(T<=1e4)', ycol, 
                xlabel=vc_label,
                ylabel='$v_\mathrm{esc}\,/\,\left[\mathrm{km\,s^{-1}}\\right]$',
                xscale='log', yscale='log',
                color='masses',
                labelsize=marker_label_size,
                xtickspace=xtickspace, 
                ytickspace=ytickspace,
                showcorr=False, adjust_text_kwargs=adjust_text_kwargs,
                arrowprops={'arrowstyle': '-'})
    vesc_hat_mw_transform = staudt_utils.log2linear(vesc_hat_mw, dvesc_mw[0]) 
    plt_forecast(ax, X_forecast * 100., np.array([[vesc_hat_mw_transform[0]]]),
                 np.array([vesc_hat_mw_transform[1:]]),
                 xadjustment=None)
    ax.annotate('Milky Way', (vc_eilers, vesc_hat_mw_transform[0]), 
                fontsize=labelsize,
                color='red', 
                arrowprops={'arrowstyle':'-|>', 'color':'red'}, 
                textcoords='axes fraction',
                xytext=(0.33, 0.66))

    # Replace the necessary data labels/annotations with their overrides
    override_labels(label_overrides, ax, dm_den.load_data(df_source),
                    ycol, 'v_dot_phihat_disc(T<=1e4)', labelsize )

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=350)
    plt.show()

    if update_values:
        # Save the vesc(Phi)-->\hat{vesc}(vc) predictions
        df = dm_den.load_data(df_source)
        if xadjustment in ['logreg_linaxunits', 'log'] \
           and yadjustment in ['logreg_linaxunits', 'log']:
            vesc_hat_dict = dict(amp * df[xcol] ** slope)
        vesc_hat_dict['mw'] = vesc_hat_mw_transform[0]
        with open(paths.data + 'vesc_hat_dict.pkl', 'wb') as f:
            pickle.dump(vesc_hat_dict, f, pickle.HIGHEST_PROTOCOL)

    return None

def plt_vs_vc(ycol, source_fname, tgt_fname=None,
              update_val=False,
              forecast_sig=1.-0.682, #forecast significance
              verbose=False, 
              adjust_text_kwargs={}, show_formula='outside',
              figsize=(10,5), labelsize=14., vc=vc_eilers, dvc=dvc_eilers,
              label_overrides={}, linear_dyfrac_data=0., show_marker_errs=None,
              **kwargs):
    '''
    Noteworthy Parameters
    ----------
    ...
    label_overrides: dict
        A dictionary for overwriting the auto-placed data labels in the form 
            {galname: (annx, anny, draw_arrow)}
        If draw_arrow is True (False), an arrow will (will not) link the data
            point with its label.
    show_marker_errs: bool, default True if ycol=='den_disc'
                                    False if ycol=='disp_dm_disc_cyl'
        Whether to show error bars on the markers. The error values come from 
        linear_y_data * linear_dyfrac_data.
    '''
    if show_marker_errs is None:
        if ycol == 'den_disc':
            show_marker_errs = True
        elif ycol == 'disp_dm_disc_cyl':
            show_marker_errs = False
    
    if len(label_overrides) > 0:
        for gal in label_overrides:
            if len(label_overrides[gal]) != 3:
                raise ValueError('`label_overrides` must be in the form '
                                 '{galname: (annx, anny, draw_arrow)}.')
    vc /= 100.
    dvc /= 100.

    df = dm_den.load_data(source_fname).drop(['m12z', 'm12w'])
    textxy = (0.04, 0.96)
    fontsize = 14
    formula_y = -0.4
    
    if 'dpi_show' in kwargs:
        dpi_show = kwargs['dpi_show']
    else:
        dpi_show = 110

    fig = plt.figure(figsize=figsize, dpi=dpi_show)

    ax = fig.add_subplot(111)
    if ycol=='den_disc':
        ylabel = den_label
        yadjustment = 'log'
        yscale = 'linear'
    elif ycol=='disp_dm_disc_cyl':
        ylabel = disp_label
        yadjustment = 'logreg_linaxunits'
        yscale = 'log'

    if 'ytickspace' not in kwargs:
        kwargs['ytickspace'] = 0.05

    x_forecast = [[vc]]
    dX = [[dvc]]
    reg_disc = ax_slr(ax,source_fname,
                      'vc100',
                      ycol,
                      xlabel=vc_label,
                      ylabel=ylabel,
                      xadjustment='logreg_linaxunits', yadjustment=yadjustment,
                      xscale='log', yscale=yscale,
                      linear_dyfrac_data=linear_dyfrac_data,
                      formula_y=formula_y, dropgals=['m12w','m12z'],
                      arrowprops={'arrowstyle':'-'}, 
                      show_formula=show_formula,
                      showlabels=True, 
                      x_forecast=x_forecast, forecast_sig=forecast_sig, 
                      dX=dX, showGeV=True, verbose=verbose,
                      adjust_text_kwargs=adjust_text_kwargs,
                      labelsize=labelsize, return_error=True, 
                      show_band=True, **kwargs)
    delta_beta = reg_disc[-3] #errors on the regression coefficients
    # The avg half width of the error band, taking the average over a range
    # of x values. 
    avg_error = reg_disc[-2] 
    # The [Ys, dYs] from the y prediction made at x_forecast:
    yhat_vc = reg_disc[-1] 

    if show_marker_errs:
        if ycol == 'den_disc':
            # log(max_y) = log(y + y * linear_dyfrac_data)
            # log(min_y) = log(y - y * linear_dyfrac_data)
            errors = np.array(
                    [[np.abs(np.log10(1. - linear_dyfrac_data))],
                     [np.log10(1. + linear_dyfrac_data)]]) 
            errors = errors.repeat(len(df), axis=1)
            ax.errorbar(df['vc100'], np.log10(df[ycol]), errors, marker='', 
                        ls='', 
                        zorder=0, c='k')
        elif ycol == 'disp_dm_disc_cyl':
            ax.errorbar(df['vc100'], df[ycol],
                        df[ycol] * linear_dyfrac_data,
                        marker='', ls='', zorder=0, c='k')

    # Set the pos of the MW label
    if ycol == 'disp_dm_disc_cyl':
        mw_text_kwargs = {'xytext':(0.33, 0.66)} 
    elif ycol == 'den_disc':
        mw_text_kwargs = {'xytext':(0.33, 0.55)}
    mw_text = ax.annotate('Milky Way', (vc, yhat_vc[0][0,0]), 
                          fontsize=labelsize,
                          color='red', 
                          arrowprops={'arrowstyle':'-|>', 'color':'red'}, 
                          textcoords='axes fraction',
                          **mw_text_kwargs)

    ###########################################################################
    y_flat = np.concatenate(yhat_vc, axis=1).flatten()

    slope_raw = reg_disc[0][0]
    dslope_raw = delta_beta[1][0] 
    slope, dslope_str = staudt_utils.sig_figs(slope_raw, dslope_raw)

    logy_intercept_raw = reg_disc[1]
    dlogy_intercept_raw = delta_beta[0][0]
    logy_intercept_str, dlogy_intercept_str = \
            staudt_utils.sig_figs(logy_intercept_raw, dlogy_intercept_raw)

    if ycol=='disp_dm_disc_cyl':
        data2save = {'disp_slope': slope_raw, 
                     'ddisp_slope': dslope_raw,
                     'logdisp_intercept': logy_intercept_raw,
                     'dlogdisp_intercept': dlogy_intercept_raw}
        dm_den.save_var_raw(data2save) 

        #y_save, dy_save = staudt_utils.log2linear(*y_flat)
        y_save, dy_save = staudt_utils.sig_figs(yhat_vc[0][0, 0],
                                                yhat_vc[1][0])
        disp_transform = staudt_utils.log2linear(logy_intercept_raw,
                                                 delta_beta[0][0])
        disp_amp = disp_transform[0]
        ddisp_amp = disp_transform[1:] 
        disp_amp_str, ddisp_amp_str = staudt_utils.sig_figs(disp_amp, 
                                                            ddisp_amp)
        if update_val:
            uci.save_prediction('disp', y_save, dy_save)
            uci.save_prediction('disp_slope', slope, dslope_str)
            uci.save_prediction('disp_amp', disp_amp_str, ddisp_amp_str)
    elif ycol=='den_disc':
        data2save = {'den_slope': slope_raw, 
                     'logden_intercept': logy_intercept_raw}
        dm_den.save_var_raw(data2save) 

        y_save, dy_save = staudt_utils.sig_figs(yhat_vc[0][0, 0],
                                                yhat_vc[1][0])

        # I expect log2linear to return asymetric errors in the following 
        # line.
        Y_MSUN = staudt_utils.log2linear(*y_flat) * u.M_sun/u.kpc**3.
        Y_1E7MSUN = Y_MSUN / 1.e7 
        y_1e7msun_txt, DY_1E7MSUN_TXT = staudt_utils.sig_figs(
                Y_1E7MSUN[0].value, Y_1E7MSUN[1:].value) 

        particle_y = (Y_MSUN * c.c**2.).to(u.GeV/u.cm**3.) 
        if len(particle_y) == 3:
            display(Latex('$\\rho_{{\\rm MW}}'
                          '={0:0.3f}^{{+{1:0.3f}}}_{{-{2:0.3f}}}'
                          '\\rm GeV\,cm^{{-3}}$'
                          .format(*particle_y.value)))
        elif len(particle_y) == 2:
            display(Latex('$\\rho_{{\\rm MW}}'
                          '={0:0.3f}\pm{1:0.3f}\\rm GeV\,cm^{{-3}}$'
                          .format(*particle_y.value)))
        else:
            raise ValueError('Unexpected number of elements in `particle_y`.')
        # The following variable assignment is written in such a way that 
        # it will work 
        # no matter whether
        # the errors are symmetric or asymetric (with the [1:])
        particle_y_save, particle_dy_save = staudt_utils.sig_figs(
               particle_y[0].value,
               particle_y[1:].value)
        
        RHO0_1E7MSUN = staudt_utils.log2linear(
                logy_intercept_raw, dlogy_intercept_raw) / 1.e7 
        RHO0_1E7MSUN *= u.M_sun / u.kpc**3.
        RHO0_GEV = (RHO0_1E7MSUN * 1.e7 * c.c**2.).to(u.GeV / u.cm**3.) 
        rho0_1e7msun_txt, DRHO0_1E7MSUN_TXT = staudt_utils.sig_figs(
                RHO0_1E7MSUN[0].value, RHO0_1E7MSUN[1:].value)
        rho0_GeV_txt, DRHO0_GEV_TXT = staudt_utils.sig_figs(
                RHO0_GEV[0].value, RHO0_GEV[1:].value)

        if update_val:
            # Save regression info 
            uci.save_prediction('den_slope', slope, dslope_str)
            uci.save_prediction('logden_intercept', logy_intercept_str, 
                                   dlogy_intercept_str)
            uci.save_prediction('rho0_1e7msun', rho0_1e7msun_txt,
                                DRHO0_1E7MSUN_TXT)

            # Save MW results
            uci.save_prediction('logrho', y_save, dy_save)
            uci.save_prediction('rho_GeV', particle_y_save, 
                                   particle_dy_save)
            uci.save_prediction('rho_1e7msun', y_1e7msun_txt, 
                                DY_1E7MSUN_TXT)
            uci.save_prediction('rho0_GeV', rho0_GeV_txt, DRHO0_GEV_TXT)
    ###########################################################################

    display(Latex('$r=8.3\pm{0:0.2f}\,\mathrm{{kpc}}$'
                  .format(df.attrs['dr']/2., df.attrs['dz']/2.)))
    display(Latex('$|z|\in[0,{1:0.2f}]\,\mathrm{{kpc}}$' \
                  .format(df.attrs['dr']/2., df.attrs['dz']/2.)))

    # Take the vc/100 values that the x-axis is based on and turn them into vc:
    ax.xaxis.set_major_formatter(lambda x, pos: '{0:0.0f}'.format(x*100.))

    # Replace the necessary labels/annotations with their overrides
    override_labels(label_overrides, ax, df, ycol, 'vc100', labelsize,
                    xadjustment=None, yadjustment=yadjustment)

    plt.draw()
    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=350)

    plt.show()
    
    return yhat_vc

def override_labels(label_overrides, ax, df, ycol, xcol, labelsize,
                    xadjustment=None, yadjustment=None):
    for child in ax.get_children():
        if isinstance(child, mpl.text.Annotation):
            text = child.get_text()
            if text in label_overrides:
                child.remove() # Delete the existing text label

                # Get the location of the data point
                data_y = df.loc[text, ycol]
                data_x = df.loc[text, xcol]
                if yadjustment == 'log':
                    data_y = np.log10(data_y)
                if xadjustment == 'log':
                    data_x = np.log10(data_x)
                point = (data_x, data_y)
                # Create the new label
                if label_overrides[text][2]: 
                    # If the override is set to draw an arrow:
                    arrowprops_override = {'arrowstyle':'-', 'shrinkB': 5}
                else:
                    arrowprops_override = None
                ax.annotate(text,
                            xy=point,
                            xytext=(label_overrides[text][0],
                                    label_overrides[text][1]),
                            arrowprops=arrowprops_override,
                            bbox=dict(pad=0., facecolor='none', ec='none'),
                            fontsize=labelsize)
        elif isinstance(child, mpl.patches.FancyArrowPatch):
            text = child.patchA.get_text()
            if text in label_overrides:
                child.remove() # Delete the existing arrow

def plt_vs_gmr_vc(ycol, tgt_fname=None, 
                  source_fname='dm_stats_dz1.0_20230626.h5',
                  forecast_sig=1.-0.682, verbose=False, 
                  adjust_text_kwargs={}, show_formula='outside',
                  figsize=(10,5), labelsize=14., vc=vc_eilers, dvc=dvc_eilers):
    df = dm_den.load_data(source_fname)

    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=110, sharey=True,
                            sharex=True)
    fig.subplots_adjust(wspace=0.)
    ax0 = axs[0]
    ax1 = axs[1]
    
    textxy = (0.04, 0.96)
    fontsize = 14
    formula_y = -0.4
    
    if ycol=='den_disc':
        ylabel = den_label
    elif ycol=='disp_dm_disc_cyl':
        ylabel = disp_label
    
    draw_shades(ax0, ycol, vc, dvc)
    draw_shades(ax1, ycol, vc, dvc)
    
    reg_gmr = ax_slr(ax0, 
                      source_fname,
                      'vcirc',
                      ycol,
                      xlabel=gmr_label, ylabel=ylabel,
                      xadjustment='log', yadjustment='log',
                      dropgals=['m12w','m12z'],
                      arrowprops={'arrowstyle':'-'}, 
                      show_formula=show_formula, x_forecast=[[vc]],
                      dX=[[dvc]], showGeV=False, 
                      showlabels=True, formula_y=formula_y, verbose=verbose,
                      adjust_text_kwargs=adjust_text_kwargs,
                      labelsize=labelsize)
    yhat_gmr = reg_gmr[-1]
    #plt_forecast(ax0, yhat_gmr)

    reg_disc = ax_slr(ax1,source_fname,
                     'v_dot_phihat_disc(T<=1e4)',
                     ycol,
                     xlabel=vc_label,
                     ylabel=ylabel,
                     xadjustment='log', yadjustment='log',
                     formula_y=formula_y, dropgals=['m12w','m12z'],
                     arrowprops={'arrowstyle':'-'}, 
                     show_formula=show_formula,
                     showlabels=True, 
                     x_forecast=[[vc]], forecast_sig=forecast_sig, 
                     dX=[[dvc]], showGeV=True, verbose=verbose,
                     adjust_text_kwargs=adjust_text_kwargs,
                     labelsize=labelsize)
    yhat_vc = reg_disc[-1]
    ax1.set_ylabel(None)
    
    display(Latex('$r=8.3\pm{0:0.2f}\,\mathrm{{kpc}}$'
                 .format(df.attrs['dr']/2., df.attrs['dz']/2.)))
    display(Latex('$|z|\in[0,{1:0.2f}]\,\mathrm{{kpc}}$' \
                 .format(df.attrs['dr']/2., df.attrs['dz']/2.)))
    #plt_forecast(ax1, yhat_vc)

    plt.draw()
    
    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=140)

    plt.show()
    
    return yhat_vc

def plt_disc_diffs(df_source, 
                   diff_source,
                   only_linear=False, 
                   only_log=False, figsize=None, tgt_fname=None,
                   update_val=False):
    '''
    For density and dispersion, plot histograms of the fractional difference
    between the average value in 
    the solar ring and that
    measured in various phi slices in that ring.

    The density and dispersion values come from a pickled dictionary that was
    created by dm_den.den_disp_phi_bins.
    '''
    direc = paths.data
    with open(direc+diff_source, 'rb') as handle:
        den_disp_dict = pickle.load(handle)
    df = dm_den.load_data(df_source)
    galnames = df.drop(['m12w','m12z']).index
    
    def setup(log):
        if not log:
            denlabel = '$\\rho(\phi)/\,\overline{\\rho}$'
            displabel = '$\sigma_\mathrm{3D}(\phi)' \
                        '/\,\overline{\sigma}_\mathrm{3D}$'

            dens = np.array([den_disp_dict[galname]['dens/avg'] \
                             for galname in galnames]).flatten()
            disps = np.array([den_disp_dict[galname]['disps/avg'] \
                              for galname in galnames]).flatten()
            den_percent_diff = staudt_utils.round_up(
                100.*np.max(np.abs(1.-dens)),
                3)
            disp_percent_diff = staudt_utils.round_up(
                100.*np.max(np.abs(1.-disps)),
                3)
            percent_diff = staudt_utils.round_up(
                100.*np.max(np.abs(1.-np.array([dens,disps]))),
                3)
            print('{0:0.2f}% max den diff'.format(den_percent_diff))
            print('{0:0.2f}% max disp diff'.format(disp_percent_diff))
            std_dens = np.std(1. - dens)
            std_disps = np.std(1. - disps)
            display(Latex(
                    '$\\rm{{st\,dev}}'
                    '{{\left(\\rho(\phi)/\overline{{\\rho}}\\right)}}'
                    '={0:0.1f}\%$'.format(std_dens * 100.)))
            display(Latex(
                    '$\\rm{{st\,dev}}'
                    '{{\left(\sigma(\phi)/\overline{{\sigma}}\\right)}}'
                    '={0:0.1f}\%$'.format(std_disps * 100.)))
            if update_val:
                dm_den.save_var_raw({'stdev_linear_dendiff': std_dens,
                                     'stdev_linear_dispdiff': std_disps})

                #update the value in data.txt for the paper
                uci.save_var_latex('maxdendiff',
                                      '{0:0.1f}\%'.format(den_percent_diff))
                uci.save_var_latex('maxdispdiff',
                                      '{0:0.1f}\%'.format(disp_percent_diff))
                uci.save_var_latex('maxdiff',
                                      '{0:0.1f}\%'.format(percent_diff))
                uci.save_var_latex('stdev_linear_dendiff',
                                      '{0:0.1f}\%'.format(std_dens * 100.))
                uci.save_var_latex('stdev_linear_dispdiff',
                                      '{0:0.1f}\%'.format(std_disps * 100.))
        else:
            denlabel = '$\log\\rho(\phi)\,/\,\log\overline{\\rho}$'
            displabel = '$\log\sigma_\mathrm{3D}(\phi)' \
                        '\,/\,\log\overline{\sigma}_\mathrm{3D}$'

            dens = np.array([den_disp_dict[galname]['log(dens)/log(avg)'] \
                             for galname in galnames]).flatten()
            disps = np.array([den_disp_dict[galname]['log(disps)/log(avg)'] \
                              for galname in galnames]).flatten()
            den_percent_diff = staudt_utils.round_up(
                100.*np.max(np.abs(1.-dens)),
                3)
            disp_percent_diff = staudt_utils.round_up(
                100.*np.max(np.abs(1.-disps)),
                3)
            percent_diff = staudt_utils.round_up(
                100.*np.max(np.abs(1.-np.array([dens,disps]))),
                3)
            print('{0:0.2f}% max log den diff'.format(den_percent_diff))
            print('{0:0.2f}% max log disp diff'.format(disp_percent_diff))
            staudt_utils.print_eq('\min\Delta\log\\rho/\log\overline{\\rho}',
                                  np.min(dens-1))
            staudt_utils.print_eq('\max\Delta\log\\rho/\log\overline{\\rho}',
                     np.max(dens-1))
            staudt_utils.print_eq(
                     '\min\Delta\log\\sigma/\log\overline{\\sigma}',
                     np.min(disps-1))
            staudt_utils.print_eq(
                     '\max\Delta\log\\sigma/\log\overline{\\sigma}',
                     np.max(disps-1))
        return denlabel, displabel, dens, disps
    
    if only_log or only_linear:
        dims = (1,2)
        if figsize is None:
            figsize = (10,4)
    else:
        dims = (2,2)
        if figsize is None:
            figsize = (10,10)
    fig, axs = plt.subplots(*dims, figsize=figsize, sharex='col', sharey='row')
    axs = axs.ravel()
    fig.subplots_adjust(wspace=0.05)
    w = 1.
    N = 10
    ec = 'w'
    ylabel = '$N_{\phi\,\mathrm{slices}}$'
    
    if not only_log:
        denlabel, displabel, dens, disps = setup(False)
        axs[0].hist(dens, N, rwidth=w, ec=ec)
        axs[0].set_xlabel(denlabel)
        axs[0].set_ylabel(ylabel)
        axs[1].hist(disps, N, rwidth=w, ec=ec)
        axs[1].set_xlabel(displabel)
    
    if not only_linear:
        denlabel, displabel, dens, disps = setup(True)
        if only_log:
            i = 0
        else:
            i = 2
        axs[i].hist(dens, N, rwidth=w, ec=ec)
        axs[i].set_xlabel(denlabel)
        axs[i].set_ylabel(ylabel)
        i += 1
        axs[i].hist(disps, N, rwidth=w, ec=ec)
        axs[i].set_xlabel(displabel)

    # Set the x tick labels to be 3 decimals
    #axs[-2].xaxis.set_major_formatter(lambda x, pos: '{0:0.3f}'.format(x))
    #axs[-1].xaxis.set_major_formatter(lambda x, pos: '{0:0.3f}'.format(x))

    if tgt_fname:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=350)

    plt.show()
    
    return None

def plt_gmr_vs_vc(df_source, tgt_fname=None,
                  figsize=(8,4),
                  labelsize=11., adjust_text_kwargs={}, label_overrides={},
                  only_disks=True):
    '''
    Parameters
    ----------
    ...
    label_overrides: dict
        A dictionary for overwriting the auto-placed data labels in the form 
            {galname: (annx, anny, draw_arrow)}
        If draw_arrow is True (False), an arrow will (will not) link the data
            point with its label.
    '''
    df = dm_den.load_data(df_source)
    if only_disks:
        df.drop(['m12z', 'm12w'], inplace=True)
    xcol = 'v_dot_phihat_disc(T<=1e4)'
    ycol = 'vcirc'

    fig = plt.figure(figsize=figsize, dpi=110)
    ax = fig.add_subplot(111)
    
    xs = (df[xcol])
    ys = (df[ycol])

    errors = (ys-xs).values
    frac_errors = errors / ys.values
    frac_std = np.sqrt((frac_errors**2.).sum() / (len(frac_errors) - 2.))
    sse = errors.T @ errors
    diffs = ys-np.mean(ys)
    tss = diffs.T @ diffs
    r2_1to1 = 1.-sse/tss
    display(Latex('$r^2_\mathrm{{1:1}}={0:0.2f}$'.format(r2_1to1)))
    display(Latex('std dev = {0:0.1f}%'.format(frac_std * 100.)))

    fill_ax_new(ax, df, xcol, ycol, 
                xlabel=vc_label, ylabel=gmr_label,
                xadjustment=None, showcorr=False, labelsize=labelsize,
                arrowprops={'arrowstyle': '-'},
                adjust_text_kwargs=adjust_text_kwargs)

    # Plot 1:1 line ###########################################################

    # The following two lines are here because I made a decision about what I
    # wanted the limits to be, but I don't always make this decision, in which
    # case I would need the ax.get_xlim and ax.get_ylim lines that follow. They
    # may seem
    # redundant in this case; however, they're there so I can copy this code
    # and use it in places where I don't first set the limits.
    #ax.set_xlim(xs.min(), xs.max())
    #ax.set_ylim(xs.min(), xs.max())

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    start = min(xlim[0], ylim[0])
    stop = max(xlim[1], ylim[1])
    ax.plot([start, stop], [start, stop], color='gray', 
            ls='--', label='1:1')
    ax.set_xlim(start, stop)
    ax.set_ylim(start, stop)
    ###########################################################################

    ax.legend(fontsize=11)

    override_labels(label_overrides, ax, df, ycol, xcol, labelsize, 
                    'log', 'log')

    if tgt_fname:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=350)

    plt.show()

    return None

def plt_particle_counts(df_source, dropgals=None):
    import cropper

    df = dm_den.load_data(df_source)
    if dropgals is not None:
        df.drop(dropgals, inplace=True)
    dz = df.attrs['dz']
    r0 = 8.3
    dr = df.attrs['drsolar']
    rmax = r0+dr/2.
    rmin = r0-dr/2.
    counts = []
    pbar = ProgressBar()
    for galname in pbar(df.index):
        gal = cropper.load_data(galname, getparts=['PartType1'], 
                                verbose=False) 
        rs = gal['PartType1']['r']
        zs = gal['PartType1']['coord_rot'][:,2]
        inshell = (rs<rmax) & (rs>rmin)
        indisc = np.abs(zs) < dz/2.
        counts += [np.sum(inshell & indisc)]
    counts = np.array(counts)
    imin = np.argmin(counts)
    imax = np.argmax(counts)
    imed = np.argpartition(counts, len(counts) // 2)[len(counts) // 2]
    for split in [1,15,30,100]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        _, bins, _ = ax.hist(counts, ec='w')
        ax.set_xticks(bins, 
                      labels=['{0:0.0f}'.format(b/split) for b in bins])
        ax.tick_params(axis='x', labelrotation=45)
        ax.grid(False)
        ax.set_ylabel('$N_\mathrm{gals}$')
        ax.set_xlabel('$N_\mathrm{DM}$ per slice')
        ax.set_title('{0:0.0f} slices'.format(split))
        plt.show()
        
        Nmin = counts.min() / split
        Nmax = counts.max() / split
        Nmed = counts[imed] / split
        print('{0:s} has the least particles: {1:0.0f}'
              .format(df.index[imin], Nmin))
        print('shot noise = {0:0.1f}%'.format(100. / np.sqrt(Nmin)))
        print('{0:s} has the most particles: {1:0.0f}'
              .format(df.index[imax], counts.max() / split))
        print('shot noise = {0:0.1f}%'.format(100. / np.sqrt(Nmax)))
        print('{0:s} has the median number of particles: {1:0.0f}'
              .format(df.index[imed], counts[imed] / split))
        print('shot noise = {0:0.1f}%'.format(100. / np.sqrt(Nmed)))
    return None

def make_sci_y(axs, i, order_of_mag):
    '''
    Put y-axis in scientific notation
    '''
    if i < 4:
        # If the plot is in the very first row, do proper scientific
        # notation so the multiplier appears at the top.
        axs[i].ticklabel_format(style='sci', axis='y', 
                                scilimits=(order_of_mag,
                                           order_of_mag),
                                useMathText=True)
    else:
        # Otherwise, just reformat the labels, because we don't want
        # multiple multiplier labels cluttering the plot.
        axs[i].yaxis.set_major_formatter(
                lambda y, pos: '{0:0.0f}'.format(y / 10.**order_of_mag))
    return None

def plt_naive(gals, vcut_type, df_source, tgt_fname=None, update_vals=False, 
              show_sigma_vc=True, show_exp=True, show_sigma_meas=True):
    '''
    Plot the naive, simple Maxwellian, with v0=vc.

    Noteworthy parameters
    ---------------------
    gals: {'discs' or list-like}
        Which glaxies to plot.
    vcut_type: {'lim_fit', 'lim', 'veschatphi', 'vesc_fit'}
        What type of cut speed to use.
            lim: The true escape speed v_esc -- The speed of the fastest DM 
                particle in 
                the solar ring.
            lim_fit: \hat{v}_{esc}(v_c) -- A regression of v_esc to vc
            veschatphi: \hat{v}_{esc}(Phi) -- An estimate of the escape speed
                based on gravitational potential Phi
            vesc_fit: \hat{v}_{esc}(Phi-->v_c) -- A regression of veschatphi
                to vc
            ideal: The final cutoff speed that would optimize the galaxy's halo
                integral's fit when calculating the halo integral of a 
                fitting.max_double_hard model

    Returns
    -------
    None
    '''
    if update_vals and gals != 'discs':
        raise ValueError('You should only update values when you\'re plotting '
                         'all the discs.')
    import dm_den
    import fitting
    with open('./data/v_pdfs_disc_dz1.0.pkl','rb') as f:
        pdfs_v=pickle.load(f)
    islist = isinstance(gals, (list, np.ndarray, 
                               pd.core.indexes.base.Index))
    df = dm_den.load_data(df_source)
    if islist:
        df = df.loc[gals]
    elif gals == 'discs':
        df.drop(['m12w', 'm12z'], inplace=True)
    else:
        raise ValueError('Unexpected value provided for gals arg')
    vesc_dict = dm_den.load_vcuts('veschatphi', df)
    vcut_dict = dm_den.load_vcuts(vcut_type, df)

    Ngals = len(df)
    if islist:
        Ncols = min(Ngals, 4)
        Nrows = math.ceil(len(gals) / Ncols)
    elif gals == 'discs':
        figsize = (19., 12.)
        Nrows = 3
        Ncols = 4
    xfigsize = 4.6 / 2. * Ncols + 1.
    yfigsize = 1.5 * Nrows + 1. 
    if Ngals == 2:
        # Add room for residual plots
        Nrows += 1
        yfigsize += 1. 
        height_ratios = [4,1]
    else:
        height_ratios = None
    fig,axs=plt.subplots(Nrows, Ncols, figsize=(xfigsize, yfigsize), 
                         sharey='row',
                         sharex=True, dpi=140, height_ratios=height_ratios)
    axs=axs.ravel()
    
    pbar = ProgressBar()

    # velocities to use when plotting the functional form attempts
    vs_maxwell = np.linspace(0., 800., 700) 
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        # Strength at which to truncate the distribution
        # I'm probably not going to use this here, though
        k = pickle.load(f)['k'] 

    rms_dict = {}
    rms_dict['sigma_vc'] = {}
    rms_dict['true_sigma'] = {}
    for i, gal in enumerate(df.index):
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        with open(paths.data + 'data_raw.pkl', 'rb') as f:
            results_dict = pickle.load(f)
        sigma_predicted = 10.**results_dict['logdisp_intercept'] \
                          * (vc/100.) ** results_dict['disp_slope']
        sigma_truth = df.loc[gal, 'disp_dm_disc_cyl']
        vcut = vcut_dict[gal]

        #######################################################################
        # Data
        #######################################################################
        bins = pdfs_v[gal]['bins']
        vs_truth = (bins[1:] + bins[:-1]) / 2.
        ps_truth = pdfs_v[gal]['ps']
        axs[i].stairs(ps_truth, bins,
                      color='k', label='data')

        #######################################################################
        # p(sigma(vc), vs_maxwell)
        #######################################################################
        ps_sigma_vc = fitting.smooth_step_max(
                                   vs_maxwell, 
                                   np.sqrt(2./3.)*sigma_predicted,
                                   np.inf,
                                   np.inf)
        rms_err_sigma_vc, _ = fitting.calc_rms_err(
                vs_truth, ps_truth, fitting.smooth_step_max,
                args=(np.sqrt(2./3)*sigma_predicted,
                      np.inf, np.inf))
        rms_sigma_vc_txt = staudt_utils.mprint(rms_err_sigma_vc, d=1, 
                                               show=False).replace('$','')
        rms_dict['sigma_vc'][gal] = rms_err_sigma_vc
        if show_sigma_vc:
            axs[i].plot(vs_maxwell, ps_sigma_vc, 
                        label='$v_0=\sigma_\mathrm{3D}(v_\mathrm{c})$')

        #######################################################################
        # p(sigma_measured, vs_maxwell)
        #######################################################################
        ps_true_sigma = fitting.smooth_step_max(vs_maxwell,
                                                np.sqrt(2./3.) * sigma_truth,
                                                np.inf,
                                                np.inf)
        rms_err_true_sigma, _ = fitting.calc_rms_err(
                vs_truth, ps_truth, fitting.smooth_step_max,
                args=(np.sqrt(2./3.)*sigma_truth,
                      np.inf, np.inf))
        rms_true_sigma_txt= staudt_utils.mprint(rms_err_true_sigma, d=1, 
                                            show=False).replace('$','')
        rms_dict['true_sigma'][gal] = rms_err_true_sigma
        if show_sigma_meas:
            axs[i].plot(vs_maxwell,
                        ps_true_sigma,
                        label = '$v_0=\sqrt{2/3}\sigma_\mathrm{meas}$')
        
        #######################################################################
        # p(vc, vs_maxwell)
        #######################################################################
        axs[i].plot(vs_maxwell, 
                    fitting.smooth_step_max(vs_maxwell, vc, np.inf, np.inf),
                    label = 'Maxwellian, $v_0=v_\mathrm{c}$')

        #######################################################################
        # p(vc, vs_maxwell) * [exponentially truncated @ vcut
        #######################################################################
        if show_exp:
            axs[i].plot(vs_maxwell,
                        fitting.exp_max(vs_maxwell, vc, vcut),
                        label = '$v_0=v_\mathrm{{c}}$'
                                '\nexp trunc @ {0:s}'
                                .format(vcut_labels[vcut_type]))
                    
        # Draw vesc line
        vesc = vesc_dict[gal]
        axs[i].axvline(vesc, ls='--', alpha=0.8, color='k')
        trans = mpl.transforms.blended_transform_factory(axs[i].transData,
                                                         axs[i].transAxes)
        if vesc >= vcut:
            vesc_adj = 20.
            vesc_ha = 'left'
            vcut_adj = 0.
            vcut_ha = 'right'
        else:
            vesc_adj = 0.
            vesc_ha = 'right'
            vcut_adj = 20. 
            vcut_ha = 'left'
        if vcut_type == 'vesc_fit':
            veschatphi_label_y = 0.8
            veschatphi_va = 'top'
            vcut_label_y = 0.8
            vcut_va = 'top'
        else:
            veschatphi_label_y = 0.4
            veschatphi_va = 'baseline'
            vcut_label_y = 0.4
            vcut_va = 'baseline'
        axs[i].text(vesc + vesc_adj, veschatphi_label_y, 
                    vcut_labels['veschatphi'], 
                    transform=trans,
                    fontsize=15., rotation=90., color='k', 
                    horizontalalignment=vesc_ha,
                    verticalalignment=veschatphi_va)

        # Draw vcut line
        #vlim = dm_den.load_vcuts('lim', df)[gal]
        #axs[i].axvline(vlim, ls='--', alpha=0.5, color='C0')
        axs[i].axvline(vcut, ls='--', alpha=0.5, color='grey')
        trans = mpl.transforms.blended_transform_factory(axs[i].transData,
                                                         axs[i].transAxes)
        axs[i].text(vcut + vcut_adj, vcut_label_y, vcut_labels[vcut_type], 
                    transform=trans,
                    fontsize=15., rotation=90., color='gray', 
                    horizontalalignment=vcut_ha, verticalalignment=vcut_va)

        axs[i].grid(False)
        # Make ticks on both sides of the x-axis:
        axs[i].tick_params(axis='x', direction='inout', length=6)
        order_of_mag = -3
        axs[i].ticklabel_format(style='sci', axis='y', 
                                scilimits=(order_of_mag,
                                           order_of_mag),
                                useMathText=True)

        loc=[0.97,0.96]
        kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                      va='top', ha='right', 
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none',
                                pad=0.))
        axs[i].annotate(gal, loc,
                        **kwargs_txt)
        if Ngals == 2:
            # Draw residual plot
            vs_resids = copy.deepcopy(vs_truth)
            vs_extend = np.linspace(vs_resids.max(), vs_maxwell.max(), 20)
            vs_resids = np.append(vs_resids, vs_extend, axis=0) 

            def calc_resids_vc():
                ps_sigma = fitting.smooth_step_max(
                        vs_truth,
                        vc,
                        np.inf,
                        np.inf)
                resids = ps_sigma - ps_truth
                inrange = (vs_truth > 75.) & (vs_truth < 175.)
                resids_extend = fitting.smooth_step_max(
                    vs_extend,
                    vc,
                    np.inf, np.inf)
                resids = np.append(resids, 
                                   resids_extend,
                                   axis=0)
                return resids
            def calc_resids(sigma):
                ps_sigma = fitting.smooth_step_max(
                        vs_truth,
                        np.sqrt(2./3.) * sigma,
                        np.inf,
                        np.inf)
                resids = ps_sigma - ps_truth
                inrange = (vs_truth > 75.) & (vs_truth < 175.)
                resids_extend = fitting.smooth_step_max(
                    vs_extend,
                    np.sqrt(2./3.) * sigma,
                    np.inf, np.inf)
                resids = np.append(resids, 
                                   resids_extend,
                                   axis=0)
                return resids
            # Remove the 0 tick label because of overlap
            y0, y1 = axs[i].get_ylim()
            visible_ticks = np.array([t for t in axs[i].get_yticks() \
                                      if t>=y0 and t<=y1])
            new_ticks = visible_ticks[visible_ticks > 0.]
            axs[i].set_yticks(new_ticks)

            axs[i+2].grid(False)
            axs[i+2].set_ylim(-resids_lim, resids_lim)

            if show_sigma_vc:
                axs[i+2].plot(vs_resids, 
                              calc_resids(sigma_predicted)/10.**order_of_mag)
            axs[i+2].axhline(0., linestyle='--', color='k', alpha=0.5,
                             lw=1.)
            axs[i+2].plot(vs_resids, 
                          calc_resids_vc()/10.**order_of_mag)
            axs[i+2].axvline(vcut, ls='--', alpha=0.5, color='grey')

            if i == 0:
                axs[i+2].set_ylabel('resids')
    if update_vals:
        fitting.save_rms_errs(rms_dict)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.,hspace=0.)
    if Ncols > 2:
        # Put the legend in the middle of the figure instead of the middle of
        # one of the axes.
        axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
                      bbox_transform=fig.transFigure, ncol=3)
    else:
        trans_legend = mpl.transforms.blended_transform_factory(
                axs[1].transAxes, fig.transFigure)
        axs[1].legend(loc='upper center', bbox_to_anchor=(0., -0.04),
                      bbox_transform=trans_legend, ncol=2)
    label_axes(axs, fig, gals)

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=250)

    plt.show()

    return None

def plt_universal_prefit(result, df_source, gals='discs', 
                         ymax=None, show_bands=True, 
                         show_sigmoid_hard=False,
                         show_sigmoid_exp=False,
                         show_max=False,
                         show_max_hard=False,
                         show_mao_prediction=False,
                         show_mao_naive=False,
                         xtickspace=None, show_rms=False,
                         tgt_path=None, scale='linear', 
                         prediction_vcut_type=None,
                         std_vcut_type=None,
                         sigmoid_damped_eqnum=None,
                         mao_eqnum=None,
                         show_plot=True,
                         samples_fname='samples_dz1.0_sigmoid_damped.h5'):
    '''
    Noteworthy parameters
    ---------------------
    prediction_vcut_type: {'lim_fit', 'lim', 'vesc_fit', 'veschatphi', 'ideal'},
               default: None
        Specifies how to determine the speed distribution cutoff for
        prediction distributions
            lim: The true escape speed v_esc -- The speed of the fastest DM 
                particle in 
                the solar ring.
            lim_fit: \hat{v}_{esc}(v_c) -- A regression of v_esc to vc
            veschatphi: \hat{v}_{esc}(Phi) -- An estimate of the escape speed
                based on gravitational potential Phi
            vesc_fit: \hat{v}_{esc}(Phi-->v_c) -- A regression of veschatphi
                to vc
            ideal: The final cutoff speed that would optimize the galaxy's halo
                integral's fit when calculating the halo integral of a 
                fitting.max_double_hard model
    std_vcut_type: {'lim_fit', 'lim', 'vesc_fit', 'veschatphi', 'ideal'},
               default: None
        Specifies how to determine the speed distribution cutoff for standard-
        assumption distributions
            lim: The true escape speed v_esc -- The speed of the fastest DM 
                particle in 
                the solar ring.
            lim_fit: \hat{v}_{esc}(v_c) -- A regression of v_esc to vc
            veschatphi: \hat{v}_{esc}(Phi) -- An estimate of the escape speed
                based on gravitational potential Phi
            vesc_fit: \hat{v}_{esc}(Phi-->v_c) -- A regression of veschatphi
                to vc
            ideal: The final cutoff speed that would optimize the galaxy's halo
                integral's fit when calculating the halo integral of a 
                fitting.max_double_hard model
    tgt_path: str, default None
        If specified, the path to which the user wants to save the resulting
        plot
    show_plot: bool, default True
        If True, display the plot.
    samples_fname: str
        The filename of the samples to use in the plot. 
    '''
    import dm_den
    import fitting
    islist = isinstance(gals, (list, np.ndarray, 
                               pd.core.indexes.base.Index))
    if not islist and gals != 'discs':
        raise ValueError('Unexpected value provided for gals arg')
    plotting_prediction = (show_mao_prediction or show_sigmoid_hard)
    if plotting_prediction and prediction_vcut_type is None:
        raise ValueError('You must specify a prediction_vcut_type if you want'
                         ' to show_mao_prediction or show_sigmoid_hard.')
    if (show_mao_naive or show_max_hard) and std_vcut_type is None:
        raise ValueError('You must specify a std_vcut_type if you want'
                         ' to show_mao_naive.')

    # I realized that this "prefit" method loads samples that were already
    # generated with ddfrac and dhfrac assumptions, so this block of code
    # doesn't do anything. I'm also now removing these kwargs from the function
    # definition.
    #if show_bands and (ddfrac is None or dhfrac is None):
    #    grid_results = grid_eval.identify()
    #    if ddfrac is None:
    #        ddfrac = grid_results[0]
    #        print('Using ddfrac = {0:0.5f}'.format(ddfrac))
    #    if dhfrac is None:
    #        dhfrac = grid_results[1]
    #        print('Using dhfrac = {0:0.5f}'.format(dhfrac))
    df = dm_den.load_data(df_source)
    with open('./data/v_pdfs_disc_dz1.0.pkl','rb') as f:
        pdfs=pickle.load(f)
    if prediction_vcut_type is not None or std_vcut_type is not None:
        if prediction_vcut_type is not None:
            prediction_vcut_dict = dm_den.load_vcuts(prediction_vcut_type, df)
        if std_vcut_type is not None:
            std_vcut_dict = dm_den.load_vcuts(std_vcut_type, df)
    if show_mao_prediction:
        with open(paths.data + 'results_mao_' + prediction_vcut_type + '.pkl', 
                  'rb') as f:
            fit_mao = pickle.load(f)
    if show_mao_naive:
        #fit_mao_naive = fitting.fit_mao_naive_aggp(std_vcut_type, df_source)
        with open(paths.data + 'data_raw.pkl', 'rb') as f:
            p_mao_naive_agg = pickle.load(f)['p_mao_naive_agg']
    if islist:
        galnames = copy.deepcopy(gals)
    elif gals == 'discs':
        pdfs.pop('m12z')
        pdfs.pop('m12w')
        galnames = pdfs.keys() 
    Ngals = len(galnames) 
    N_postfit = 300
    vs_postfit = np.linspace(0., 700., N_postfit)
    
    if type(result) in [lmfit.model.ModelResult, 
                        lmfit.minimizer.MinimizerResult]:
        d, e, h, j, k = [result.params[key] 
                         for key in ['d', 'e', 'h', 'j', 'k']]
    elif type(result) == dict:
        d, e, h, j, k = [result[key]
                         for key in ['d', 'e', 'h', 'j', 'k']]
    else:
        raise ValueError('Unexpected data type provided for `params`')
    samples = fitting.load_samples(samples_fname)

    fig, axs = setup_multigal_fig(gals)

    sse_mao_full = 0. # SSE for the fully predictive Mao model
    sse_mao_naive = 0. # SSE for Mao using v0=vc
    sse_staudt = 0. # SSE for our predictive model
    sse_sigmoid_hard = 0. # SSE for our predictive model with a hard final cut
    sse_max_hard = 0. # SSE for the maxwellian with a hard card
    N_data = 0. # Number of truth datapoints evaluated, for agg RMS calculation

    pbar = ProgressBar()
    O_rms = -3. # Order of magnite for RMS shows in the plots
    for i, gal in enumerate(pbar(galnames)):
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        if prediction_vcut_type is not None:
            prediction_vcut = prediction_vcut_dict[gal]
        if std_vcut_type is not None:
            std_vcut = std_vcut_dict[gal]
        v0 = d * (vc / 100.) ** e
        vdamp = h * (vc / 100.) ** j
        ps_postfit = fitting.smooth_step_max(vs_postfit,
                                             v0, vdamp,
                                             k)
        if show_rms:
            vs_truth = pdfs[gal]['vs']
            N_data += len(vs_truth)
            ps_truth = pdfs[gal]['ps']
            rms_staudt, sse_staudt_add = fitting.calc_rms_err(
                    vs_truth, ps_truth,
                    fitting.smooth_step_max,
                    args=[v0, vdamp, k])
            sse_staudt += sse_staudt_add

        # Error bands
        if show_bands:
            samples_color = plt.cm.viridis(0.5)
            lowers, uppers = fitting.gal_bands_from_samples(samples['vs'],
                                                            samples[gal],
                                                            samples_color,
                                                            axs[i])
            band = np.array([lowers, uppers])
            percent_diffs = np.abs(
                    -1. + band / fitting.smooth_step_max(
                                            samples['vs'],
                                            v0, 
                                            vdamp, 
                                            k)
            ).max(axis=0)
            print('{0:s}'.format(gal))
            print(*np.array([samples['vs'], percent_diffs, lowers, uppers]).T,
                  sep='\n')
            print('')

            axs[i].fill_between(
                    samples['vs'],
                    lowers, 
                    uppers, 
                    color=plt.cm.viridis(1.), 
                    alpha=0.9, 
                    ec=samples_color, 
                    zorder=1, 
                    label='$1\sigma$ band'
            )

        # Plot data
        axs[i].stairs(pdfs[gal]['ps'], pdfs[gal]['bins'], color='k',
                      label='data')
        # Plot prediction
        if sigmoid_damped_eqnum is not None:
            sigmoid_damped_label = 'Eqn. {0:s}' \
                                   .format(str(sigmoid_damped_eqnum))
        else:
            sigmoid_damped_label = 'prediction from $v_\mathrm{c}$'
        axs[i].plot(vs_postfit,
                    ps_postfit,
                    '-',
                    label=sigmoid_damped_label, color='C3', lw=1.5, zorder=10)
        
        # Plot the prediction with a hard cut @ prediction_vcut
        if show_sigmoid_hard:
            axs[i].plot(vs_postfit,
                        fitting.max_double_hard(vs_postfit, v0, vdamp, k,
                                                prediction_vcut),
                        label=('prediction, hard cut @ '
                               + vcut_labels[prediction_vcut_type]))
            if show_rms:
                _, sse_sigmoid_hard_add = fitting.calc_rms_err(
                    vs_truth, ps_truth, fitting.max_double_hard,
                    args=(v0, vdamp, k, prediction_vcut))
                sse_sigmoid_hard += sse_sigmoid_hard_add

        # Plot the prediction with an exponential cut @ vlim_fit
        if show_sigmoid_exp:
            with open(paths.data + 'params_sigmoid_exp.pkl', 'rb') as f:
                params_sigmoid_exp = pickle.load(f)
            d_exp = params_sigmoid_exp['d']
            e_exp = params_sigmoid_exp['e']
            h_exp = params_sigmoid_exp['h']
            j_exp = params_sigmoid_exp['j']
            k_exp = params_sigmoid_exp['k']
            v0_exp = d_exp * (vc / 100.) ** e_exp
            vdamp_exp = h_exp * (vc / 100.) ** j_exp
            vlim_fit = dm_den.load_vcuts('lim_fit', df)[gal]
            axs[i].plot(vs_postfit,
                        fitting.max_double_exp(vs_postfit,
                                               v0_exp, vdamp_exp, k_exp,
                                               vlim_fit),
                        label=('prediction, exp cut @ ' 
                               + vcut_labels['lim_fit']))

        if show_max:
            axs[i].plot(
                vs_postfit,
                fitting.smooth_step_max(
                    vs_postfit,
                    vc,
                    np.inf,
                    np.inf
                ),
                label='Maxwellian, $v_0=v_{\\rm c}$', color=max_naive_color
            )

        if show_max_hard:
            axs[i].plot(vs_postfit, 
                        fitting.smooth_step_max(vs_postfit,
                                                vc, std_vcut, np.inf),
                        label=('Maxwellian, $v_0=v_{\\rm c}$, cut @ ' 
                               + vcut_labels[std_vcut_type]))
            if show_rms:
                rms_max, sse_max_hard_add = fitting.calc_rms_err(
                        vs_truth, ps_truth,
                        fitting.smooth_step_max,
                        args=(vc, std_vcut, np.inf))
                sse_max_hard += sse_max_hard_add

        if show_mao_prediction:
            v0_mao = fit_mao['d'] * (vc / 100.) ** fit_mao['e'],
            if mao_eqnum is not None:
                mao_prediction_label = 'Eqn. {0:0.0f} w/our method' \
                                       .format(mao_eqnum)
            else:
                mao_prediction_label = 'Mao w/our method'
            axs[i].plot(vs_postfit,
                        fitting.mao(vs_postfit, 
                                    v0_mao,
                                    prediction_vcut,
                                    fit_mao['p']),
                        label=mao_prediction_label,
                        color=mao_prediction_color)
            if show_rms:
                rms_mao, sse_mao_full_add = fitting.calc_rms_err(
                        vs_truth, ps_truth,
                        fitting.mao,
                        args=[v0_mao,
                              prediction_vcut,
                              fit_mao['p']])
                sse_mao_full += sse_mao_full_add
                                                                 
        if show_mao_naive:
            if mao_eqnum is not None:
                mao_naive_label = 'Eqn. {0:0.0f}, $v_0=v_\mathrm{{c}}$' \
                                  .format(mao_eqnum)
            else:
                mao_naive_label = 'Mao, $v_0=v_\mathrm{c}$'
            axs[i].plot(vs_postfit,
                        fitting.mao(vs_postfit,
                                    vc, std_vcut, 
                                    p_mao_naive_agg),
                        label=mao_naive_label, color=mao_naive_color)
            rms_mao_naive, sse_mao_naive_add = fitting.calc_rms_err(
                    vs_truth, ps_truth,
                    fitting.mao,
                    args=(vc, std_vcut, p_mao_naive_agg))
            sse_mao_naive += sse_mao_naive_add
        # Make ticks on both sides of the x-axis:
        axs[i].tick_params(axis='x', direction='inout', length=6)

        order_of_mag = -3
        if scale == 'linear':
            make_sci_y(axs, i, order_of_mag)

        if Ngals < 5:
            # Remove the 0 tick label because of overlap
            y0, y1 = axs[i].get_ylim()
            visible_ticks = np.array([t for t in axs[i].get_yticks() \
                                      if t>=y0 and t<=y1])
            new_ticks = visible_ticks[visible_ticks > 0.]
            axs[i].set_yticks(new_ticks)

            # Draw residual plot
            vs_resids = copy.deepcopy(pdfs[gal]['vs'])
            vs_extend = np.linspace(vs_resids.max(), vs_postfit.max(), 20)
            vs_resids = np.append(vs_resids, vs_extend, axis=0)
            ps_hat = fitting.smooth_step_max(pdfs[gal]['vs'], v0, vdamp, k)
            resids = ps_hat - pdfs[gal]['ps']
            resids_extend = fitting.smooth_step_max(vs_extend, v0, vdamp, k)
            resids = np.append(resids, resids_extend, axis=0)
            axs[i + Ngals].plot(vs_resids, resids / 10.**order_of_mag, 
                                color='C3')
            axs[i + Ngals].axhline(0., linestyle='--', color='k', alpha=0.5, 
                                   lw=1.)
            axs[i + Ngals].set_ylim(-resids_lim, resids_lim)
            if i == 0:
                axs[i + Ngals].set_ylabel('resids')
        loc = [0.97,0.95]
        if Ngals == 12:
            namefs = 13. #Font size for galaxy name
            detail_fontsize = 8. #Font size for metrics shown on panels
            spacing = 0.16
        else:
            namefs = 16. #Font size for galaxy name
            detail_fontsize = 11. #Font size for metrics shown on panels 
            spacing = 0.12
        kwargs_txt = dict(fontsize=namefs, xycoords='axes fraction',
                          va='top', ha='right',
                          bbox=dict(facecolor='white', alpha=0.8, 
                                    edgecolor='none', pad=0.))
        axs[i].annotate(gal, loc,
                        **kwargs_txt)
        if show_rms:
            loc[1] -= spacing
            kwargs_txt['fontsize'] = detail_fontsize 
            if show_max_hard:
                rms_txt_max = '\nRMS$_{{\\rm Max}}={4:0.2f}$'
            else:
                rms_txt_max = ''
            if show_mao_naive:
                rms_txt_mao_naive = '\nRMS$_{{\mathrm{{M}}v_\mathrm{{c}}}}' \
                                    '={3:0.2f}$'
            else:
                rms_txt_mao_naive = ''
            if show_mao_prediction:
                rms_txt_mao = '\nRMS$_\mathrm{{Mao}}={2:0.2f}$' 
            else:
                rms_txt_mao = ''
            txt_rms = (#'\n$v_\mathrm{{c}}={0:0.0f}\,\mathrm{{km\,s^{{-1}}}}$'
                       'RMS$_{{{5:s}}}={1:0.2f}$' 
                       + rms_txt_mao_naive
                       + rms_txt_mao
                       + rms_txt_max)
            axs[i].annotate(
                txt_rms.format(
                    vc, 
                    rms_staudt / 10. ** O_rms,
                    rms_mao / 10. ** O_rms if show_mao_prediction else None,
                    rms_mao_naive / 10. ** O_rms if show_mao_naive else None,
                    rms_max / 10. ** O_rms if show_max_hard else None,
                    str(sigmoid_damped_eqnum)
                ),
                loc, zorder=0, **kwargs_txt)
        axs[i].grid(False)
        if ymax is not None:
            axs[i].set_ylim(top=ymax)

        if xtickspace is not None:
            axs[i].xaxis.set_major_locator(
                    mpl.ticker.MultipleLocator(base=xtickspace))
            axs[i].xaxis.set_minor_locator(plt.NullLocator())

        if scale != 'linear':
            axs[i].set_yscale(scale)
    print('Done plotting galaxies. Finalizing figure.')

    label_axes(axs, fig, gals)
    if fig.Nrows == 3:
        ncol = 4 
        legend_y = 0.03
    else:
        ncol = 2
        legend_y = 0.
    handles, labels = axs[0].get_legend_handles_labels()
    if show_bands:
        print('Plotting bands.')
        handles.append(mpl.lines.Line2D([0], [0], color=samples_color, lw=1.,
                                        label='rand samples'))
    axs[0].legend(handles=handles,
                  bbox_to_anchor=(0.5, legend_y), 
                  loc='upper center', ncol=ncol,
                  bbox_transform=fig.transFigure,
                  borderaxespad=1.5)

    if tgt_path is not None:
        plt.savefig(tgt_path,
                    bbox_inches='tight',
                    dpi=350)
    if show_plot:
        plt.show()
    else:
        plt.close()

    if show_rms:
        d = 2
        if show_mao_prediction:
            txt = staudt_utils.mprint(
                    np.sqrt(sse_mao_full / N_data),
                    d=d, 
                    show=False).replace('$','')
            display(Latex('$\mathrm{{RMS_{{Mao, prediction}}}}={0:s}$'
                          .format(txt)))
        if show_mao_naive:
            txt = staudt_utils.mprint(np.sqrt(sse_mao_naive / N_data),
                                      d=d, show=False).replace('$', '')
            display(Latex('$\mathrm{{RMS}}_{{\mathrm{{Mao}},v_\mathrm{{c}}}}'
                          '={0:s}$'
                          .format(txt)))
        if show_max_hard:
            txt = staudt_utils.mprint(np.sqrt(sse_max_hard / N_data),
                                      d=d, show=False).replace('$', '')
            display(Latex('$\mathrm{{RMS}}_{{\mathrm{{Max}}}}={0:s}$'
                          .format(txt)))

        txt = staudt_utils.mprint(
                np.sqrt(sse_staudt / N_data),
                d=d, 
                show=False).replace('$','')
        display(Latex('$\mathrm{{RMS_{{Staudt, prediction}}}}={0:s}$' 
                      .format(txt)))

        if show_sigmoid_hard:
            txt = staudt_utils.mprint(np.sqrt(sse_sigmoid_hard / N_data),
                                      d=d, show=False).replace('$', '')
            display(Latex('$\mathrm{{RMS_{{Sigmoid, hard}}}}={0:s}$'
                          .format(txt)))

    return None

def plt_mao_bands(dfsource):
    import dm_den
    import fitting

    with open(paths.data + 'results_mao_lim_fit.pkl', 'rb') as f:
        params = pickle.load(f)
    with open('./data/v_pdfs_disc_dz1.0.pkl','rb') as f:
        pdfs = pickle.load(f)
    df = dm_den.load_data(dfsource)
    vcut_dict = dm_den.load_vcuts('lim_fit', df)
    samples = fitting.load_samples('samples_dz1.0_mao.h5')

    ddfrac, dpfrac = grid_eval_mao.identify('grid_mao.h5')
    print('using (ddfrac, dpfrac) = ({0:0.3f}, {1:0.3f})'.format(ddfrac, 
                                                                 dpfrac))

    pdfs.pop('m12z')
    pdfs.pop('m12w')
    galnames = pdfs.keys() 
    Ngals = len(galnames)

    fig, axs = setup_multigal_fig('discs', False)
    pbar = ProgressBar()
    for i, gal in enumerate(pbar(galnames)):
        vc100 = df.loc[gal, 'vc100']
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        vcut = vcut_dict[gal]
        v0 = params['d'] * vc100 ** params['e']
        vs_postfit = np.linspace(0., 700., 300)

        samples_color = plt.cm.viridis(0.5)
        lowers, uppers = fitting.gal_bands_from_samples(
                samples['vs'], samples[gal],
                samples_color=samples_color, ax=axs[i])
        axs[i].fill_between(samples['vs'],
                            lowers, uppers, 
                            color=plt.cm.viridis(1.), 
                            alpha=0.9, 
                            ec=samples_color, zorder=1, 
                            label='$1\sigma$ band')

        axs[i].stairs(pdfs[gal]['ps'], pdfs[gal]['bins'], color='k')
        axs[i].plot(vs_postfit, fitting.mao(vs_postfit, v0, vcut,
                                            params['p']),
                    c=mao_prediction_color)

        axs[i].set_ylim(0., 0.006)

        if Ngals == 12:
            namefs = 13. #Font size for galaxy name
            detail_fontsize = 8. #Font size for metrics shown on panels
            spacing = 0.16
        else:
            namefs = 16. #Font size for galaxy name
            detail_fontsize = 11. #Font size for metrics shown on panels 
            spacing = 0.12
        kwargs_txt = dict(fontsize=namefs, xycoords='axes fraction',
                          va='top', ha='right',
                          bbox=dict(facecolor='white', alpha=0.8, 
                                    edgecolor='none', pad=0.))
        loc = [0.97,0.95]
        axs[i].annotate(gal, loc,
                        **kwargs_txt)

    plt.show()
    return None

def plt_mw(vcut_type, tgt_fname=None, dvc=0., dpi=140, show_vcrit=False,
           sigmoid_damped_eqnum=None, show_vc=False, show_current_params=True,
           show_shm=False):
    '''
    Parameters
    ----------
    vcut_type: {'lim_fit', 'lim', 'vesc_fit', 'veschatphi', 'ideal'},
               default 'lim_fit'
        Specifies how to determine the speed distribution cutoff.
            lim: The true escape speed v_esc -- The speed of the fastest DM 
                particle in 
                the solar ring.
            lim_fit: \hat{v}_{esc}(v_c) -- A regression of v_esc to vc
            veschatphi: \hat{v}_{esc}(Phi) -- An estimate of the escape speed
                based on gravitational potential Phi
            vesc_fit: \hat{v}_{esc}(Phi-->v_c) -- A regression of veschatphi
                to vc
            ideal: The final cutoff speed that would optimize the galaxy's halo
                integral's fit when calculating the halo integral of a 
                fitting.max_double_hard model
    tgt_fname: str
        File name of the plot image to save.
    dvc: float
        Uncertainty in the MW's circular speed.
    dpi: int
        Dots per inch for the displayed figure.
    show_vcrit: bool
        Whether to show a vertical line where we determine this work's speed
        distribution prediction makes its final drop below the standard
        assumption (with a hard cut at vesc_hat(vc)).
    sigmoid_damped_equm: int, default None
        The equation number of our final model in the LaTeX paper
    show_vc: bool, default False
        Whether to annotate the circular velocity under the "Milky Way" title
    show_current_params: bool, default True
        Whether to show a Maxwellian with vc from eithers and vesc(vc)
    show_shm: bool, default False
        Whether to show a Maxwellian with recommended parameters from Baxter et
        al. 2021 (v0 = vc = 238 km/s, vesc = 544 km/s)

    Returns
    -------
    None
    '''
    import grid_eval
    import dm_den
    import fitting
    df = staudt_tools.init_df()
    df.loc['mw', 'vesc'] = vesc_mw
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        results = pickle.load(f)
    ddfrac, dhfrac = grid_eval.identify()

    vc = vc_eilers
    vs = np.linspace(0., 700., 300)

    def predict(vc, ax, **kwargs):
        df = dm_den.load_data('dm_stats_20221208.h5')
        df.loc['mw', 'v_dot_phihat_disc(T<=1e4)'] = vc
        v0 = results['d'] * (vc / 100.) ** results['e']
        vdamp = results['h'] * (vc / 100.) ** results['j']
        ps = fitting.smooth_step_max(vs, v0, vdamp, results['k'])
        if sigmoid_damped_eqnum is not None:
            label = 'Eqn. {0:s}'.format(str(sigmoid_damped_eqnum))
        else:
            label = 'prediction from $v_\mathrm{c}$'
        ax.plot(vs, ps, label=label,
                **kwargs)
        lowers, uppers = fitting.gal_bands('mw', vs, df, results, ddfrac, 
                                           dhfrac, 
                                           ax=None, dvc=dvc)
        ax.fill_between(vs, lowers, uppers, 
                        alpha=0.7, 
                        lw=0.,
                        #color='#c0c0c0',
                        color='pink',
                        zorder=1, 
                        label='$1\sigma$ band')
        return None
    
    fig = plt.figure(figsize = (5., 2.5), dpi=dpi,
                     facecolor = (1., 1., 1., 0.))
    ax = fig.add_subplot(111)

    vesc_hat_dict = dm_den.load_vcuts(vcut_type, df)
    if show_current_params:
        ax.plot(vs, fitting.smooth_step_max(vs, vc, vesc_hat_dict['mw'], 
                                            np.inf),
                ls='--',
                label='Maxwellian,\n$v_0=v_\mathrm{c}$')

    if show_shm:
        ax.plot(vs, fitting.smooth_step_max(vs, v0_std, vesc_std, np.inf),
                ls='--',
                label='SHM')

    #ax.plot(vs, fitting.exp_max(vs, vc, vesc_hat_dict['mw']))
    
    predict(vc, ax, c='C3')

    if show_vcrit:
        with open(paths.data + 'vcrits_fr_distrib.pkl', 'rb') as f:
            vcrits = pickle.load(f)
        ax.axvline(vcrits['mw'], ls='--', color='grey', alpha=0.8)
    ax.set_ylabel('$f(v)\,4\pi v^2\,/\,[\mathrm{km^{-1}\,s}]$')
    ax.set_xlabel('$v\,/\,[\mathrm{km\,s^{-1}}]$')
    ax.set_ylim(0., None)
    loc = [0.97,0.96]
    kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                      va='top', ha='right',
                      bbox=dict(facecolor='white', alpha=0.8, 
                                edgecolor='none', pad=0.1))
    ax.annotate('Milky Way', loc,
                **kwargs_txt)
    loc[1] -= 0.15
    if show_vc:
        if dvc == 0.:
            kwargs_txt['fontsize'] = 11.
            ax.annotate('$v_\mathrm{{c}}={0:0.0f}\,\mathrm{{km\,s^{{-1}}}}$'
                        .format(vc),
                        loc, **kwargs_txt)
        else:
            kwargs_txt['fontsize'] = 9.
            ax.annotate('$v_\mathrm{{c}}={0:0.0f}\pm{1:0.0f}'
                        '\,\mathrm{{km\,s^{{-1}}}}$'
                        .format(vc, dvc),
                        loc, **kwargs_txt)

    # Put y-axis in scientific notation
    order_of_mag = -3
    ax.ticklabel_format(style='sci', axis='y', 
                        scilimits=(order_of_mag,
                                   order_of_mag),
                            useMathText=True)
    ax.legend(bbox_to_anchor=(1., 0.5), 
              loc='center left', ncol=1,
              bbox_transform=ax.transAxes,
              borderaxespad=1.)

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=350)

    plt.show()

    return None

def plt_halo_integrals(gals, 
                       df_source,
                       show_sigmoid_hard=False, show_sigmoid_exp=False,
                       show_max_hard=False, show_max_exp=False,
                       show_mao_prediction=False,
                       show_mao_naive=False,
                       show_std_vcut=False, show_prediction_vcut=False,
                       show_vcrit=False,
                       std_vcut_type=None, prediction_vcut_type=None,
                       xmax=None,
                       ymin=1.e-6, ymax=None,
                       xtickspace=None, 
                       scale='log',
                       tgt_fname=None, show_rms=False,
                       sigmoid_damped_eqnum=None):
    '''
    Noteworthy parameters
    ---------------------
    show_std_vcut: bool
        Whether to show a vertical line at the location of the vcut we use for
        standard-assumption/naive distributions
    show_prediction_vcut: bool
        Whether to show a vertical line at the location of the vcut we use for
        prediction distributions
    show_vcrit: bool
        Whether to draw a vertical line where this work's prediction for the
        speed distribution (not the halo integral) makes its final drop beneath
        the standard Maxwellian assumption (v0=vc, cut @ vesc_hat(vc))
    std_vcut_type: {'lim_fit', 'lim', 'vesc_fit', 'veschatphi', 'ideal'},
                   default: None
        Specifies how to determine the speed distribution cutoff for standard
        assumption distributions like the standard Maxwellian and the naive 
        Mao,
        where v0=vc for both.
            lim: The true escape speed v_esc -- The speed of the fastest DM 
                particle in 
                the solar ring.
            lim_fit: \hat{v}_{esc}(v_c) -- A regression of v_esc to vc
            veschatphi: \hat{v}_{esc}(Phi) -- An estimate of the escape speed
                based on gravitational potential Phi
            vesc_fit: \hat{v}_{esc}(Phi-->v_c) -- A regression of veschatphi
                to vc
            ideal: The final cutoff speed that would optimize the galaxy's halo
                integral's fit when calculating the halo integral of a 
                fitting.max_double_hard model
    prediction_vcut_type: {'lim_fit', 'lim', 'vesc_fit', 'veschatphi', 'ideal'},
                          default: None
        Specifies how to determine the speed distribution cutoff for prediction
        distributions like the universally fit damped sigmoid and Mao.
            lim: The true escape speed v_esc -- The speed of the fastest DM 
                particle in 
                the solar ring.
            lim_fit: \hat{v}_{esc}(v_c) -- A regression of v_esc to vc
            veschatphi: \hat{v}_{esc}(Phi) -- An estimate of the escape speed
                based on gravitational potential Phi
            vesc_fit: \hat{v}_{esc}(Phi-->v_c) -- A regression of veschatphi
                to vc
            ideal: The final cutoff speed that would optimize the galaxy's halo
                integral's fit when calculating the halo integral of a 
                fitting.max_double_hard model
    ymax: float; default: 9.e-3 for log scale, None for linear scale
        Upper limit for the y-axis. 
    '''
    import dm_den
    import fitting
    if gals != 'discs' and not isinstance(gals, (list, np.ndarray)):
        raise ValueError('Unexpected value provided for gals arg')
    plotting_a_cut_prediction = (show_sigmoid_hard 
                                 or show_mao_prediction)
    if plotting_a_cut_prediction and prediction_vcut_type is None:
        raise ValueError('You must specify a prediction_vcut_type if you want'
                         ' to show_mao_prediction, or show_sigmoid_hard.')
    plotting_a_cut_std = (show_max_hard or show_max_exp or show_mao_naive)
    if plotting_a_cut_std and std_vcut_type is None:
        raise ValueError('You must specify a std_vcut_type if you want to'
                         ' show_max_hard, show_max_exp, or show_mao_naive.')
    if show_std_vcut and std_vcut_type is None:
        raise ValueError('You must specify a std_vcut_type if you want to'
                         ' show_std_vcut.')
    if show_prediction_vcut and prediction_vcut_type is None:
        raise ValueError('You must specify a prediction_vcut_type if you want'
                         ' to show_prediction_vcut.')
    if ymax is None and scale == 'log':
        ymax = 9.e-3
    df = dm_den.load_data(df_source)

    if std_vcut_type is not None:
        std_vcut_dict = dm_den.load_vcuts(std_vcut_type, df)
        if std_vcut_type == 'veschatphi':
            std_vcut_dict['mw'] = vesc_mw
    if prediction_vcut_type is not None:
        prediction_vcut_dict = dm_den.load_vcuts(prediction_vcut_type, df)

    if gals == ['mw']:
        df.loc['mw', 'v_dot_phihat_disc(T<=1e4)'] = vc_eilers
        df.loc['mw', 'vc100'] = vc_eilers / 100.
    with open('./data/v_pdfs_disc_dz1.0.pkl','rb') as f:
        pdfs=pickle.load(f)
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        params = pickle.load(f)
    with open('./data/vcrits_fr_distrib.pkl', 'rb') as f:
        vcrits = pickle.load(f)
    if gals == 'discs':
        gal_names = list(pdfs.keys())
        for gal in ['m12z', 'm12w']:
            gal_names.remove(gal)
    else:
        gal_names = gals.copy()

    if show_mao_naive:
        fit_mao_naive = fitting.fit_mao_naive_aggp(std_vcut_type, df_source)

    fig, axs = setup_multigal_fig(gals, show_resids=False) 

    # Setting some figure parameters that should differ based on how big the
    # figure is / how many galaxies we're showing
    if fig.Ncols > 3:
        vesc_fs = 12.
        vesc_y = 0.4
        legend_ncols = 3 
    elif fig.Ncols == 1:
        legend_ncols = 1
    else:
        vesc_fs = 15.
        vesc_y = 0.45
        legend_ncols = 2

    pbar = ProgressBar()
    for i, gal in enumerate(pbar(gal_names)):
        vc100 = df.loc[gal, 'vc100']
        if std_vcut_type is not None:
            std_vcut = std_vcut_dict[gal]
        if prediction_vcut_type is not None:
            prediction_vcut = prediction_vcut_dict[gal]

        # Plot data
        if gal != 'mw':
            vs_truth = pdfs[gal]['vs']
            gs = fitting.numeric_halo_integral(pdfs[gal]['bins'], 
                                               pdfs[gal]['ps'])
            axs[i].stairs(gs, pdfs[gal]['bins'], color='k',
                          label='data', baseline=None)

        # Plot the prediction
        vs_hat = np.linspace(0., 820., 400)
        v0 = params['d'] * vc100 ** params['e']
        vdamp = params['h'] * vc100 ** params['j']
        gs_hat = fitting.g_smooth_step_max(vs_hat, v0, vdamp, params['k'])
        if sigmoid_damped_eqnum is not None:
            sigmoid_damped_label = '$f(v)=\mathrm{{Eqn.}}\,{0:s}$' \
                                   .format(str(sigmoid_damped_eqnum))
        else:
            sigmoid_damped_label = 'prediction from $v_\mathrm{c}$'
        axs[i].plot(vs_hat, gs_hat, label=sigmoid_damped_label,
                    color='C3', zorder=51)
        if show_rms:
            rms_staudt, _ = fitting.calc_rms_err(vs_truth, gs, 
                                                 fitting.g_smooth_step_max,
                                                 (v0, vdamp, params['k']))

        # Plot the prediction with an additional exponential cutoff 
        # reaching
        # 0 at vesc
        if show_sigmoid_exp:
            with open(paths.data + 'params_sigmoid_exp.pkl', 'rb') as f:
                params_sigmoid_exp = pickle.load(f)
            d_exp = params_sigmoid_exp['d']
            e_exp = params_sigmoid_exp['e']
            h_exp = params_sigmoid_exp['h']
            j_exp = params_sigmoid_exp['j']
            k_exp = params_sigmoid_exp['k']
            v0_exp = d_exp * vc100 ** e_exp
            vdamp_exp = h_exp * vc100 ** j_exp
            vlim_fit = dm_den.load_vcuts('lim_fit', df)[gal]
            axs[i].plot(vs_hat, 
                        fitting.calc_g_general(vs_hat,
                                               fitting.pN_max_double_exp,
                                               (v0_exp, vdamp_exp, k_exp,
                                                vlim_fit)),
                        label='prediction from $v_\mathrm{c}$'
                              ', exp cut @ ' + vcut_labels['lim_fit'], 
                        color='C3', ls=':', zorder=10)
            
        # Plot the prediction with a final hard cutoff at vesc
        if show_sigmoid_hard:
            axs[i].plot(vs_hat, 
                        fitting.calc_g_general(vs_hat,
                                               fitting.pN_max_double_hard,
                                               (v0, vdamp, params['k'], 
                                                prediction_vcut)),
                        #label='prediction from $v_\mathrm{c}$'
                        #      ', cut @ ' + vcut_labels[prediction_vcut_type], 
                        color='C3', ls='--', zorder=10)

        # Plot simple maxwellian with v0 = vc
        gs_max = fitting.g_smooth_step_max(vs_hat, vc100 * 100.,
                                           np.inf, np.inf)
        axs[i].plot(vs_hat,
                    gs_max,
                    label='Maxwellian$,\,v_0=v_\mathrm{c}$',
                    color='C0', zorder=50)
        
        # Plot Maxwellian with v0 = vc and an exponential cutoff at vesc
        if show_max_exp:
            axs[i].plot(vs_hat,
                        fitting.g_exp(vs_hat, vc100 * 100., std_vcut),
                        color='C0', ls=':',
                        label=('$v_0=v_\mathrm{c}$, exp cut @ ' 
                               + vcut_labels[std_vcut_type]))

        # Plot maxwellian with v0 = vc, hard truncation @ vesc
        if show_max_hard:
            axs[i].plot(vs_hat,
                        fitting.g_smooth_step_max(vs_hat, vc100 * 100.,
                                                  std_vcut, np.inf),
                        #label=('$v_0=v\mathrm{c}$, cut @ ' 
                        #       + vcut_labels[std_vcut_type]),
                        color='C0', ls='--' )

        if show_mao_prediction or show_mao_naive:
            with open('./data/results_mao_' + prediction_vcut_type + '.pkl', 
                      'rb') as f:
                results_mao = pickle.load(f)
        if show_mao_prediction:
            # Plot the halo integral resulting from a universal Mao fit
            v0_mao = results_mao['d'] * vc100 ** results_mao['e']
            axs[i].plot(vs_hat,
                        fitting.calc_g_general(vs_hat,
                                               fitting.pN_mao,
                                               (v0_mao, prediction_vcut, 
                                                results_mao['p'])),
                        label='Mao w/our method',
                        color=mao_prediction_color)
        if show_mao_naive:
            # Plot the halo integral from using v0=vc with Mao
            axs[i].plot(vs_hat,
                        fitting.calc_g_general(
                            vs_hat,
                            fitting.pN_mao,
                            (vc100 * 100., std_vcut,
                            fit_mao_naive.params['p'].value)),
                        label='Mao, $v_0=v_\mathrm{c}$',
                        color=mao_naive_color)
            if show_rms:
                rms_mao_naive, _ = fitting.calc_rms_err(
                        vs_truth, gs,
                        fitting.calc_g_general,
                        (fitting.pN_mao,
                         (vc100 * 100., 
                          std_vcut,
                          results_mao['p'])))

        loc = [0.97,0.95]
        if fig.Nrows == 3:
            namefs = 13.
        else:
            namefs = 16. 
        kwargs_txt = dict(fontsize=namefs, xycoords='axes fraction',
                          va='top', ha='right',
                          bbox=dict(facecolor='white', alpha=0.8, 
                                    edgecolor='none'))
        if gal == 'mw':
            galtxt = 'Milky Way'
        else:
            galtxt = gal
        axs[i].annotate(galtxt, loc,
                        **kwargs_txt)
        if show_rms:
            step = 0.15
            loc[1] -= step 
            kwargs_txt['fontsize'] = 9.
            axs[i].annotate('RMS$_\mathrm{{Staudt}}={0:0.2e}$'.format(
                                rms_staudt),
                            loc, **kwargs_txt)
            if show_mao_naive:
                loc[1] -= step
                axs[i].annotate('RMS$_\mathrm{{Mao}}={0:0.2e}$'.format(
                                    rms_mao_naive),
                                loc, **kwargs_txt)
        if show_std_vcut:
            # Draw vesc line
            axs[i].axvline(std_vcut, ls='--', c='grey', alpha=0.5)
            trans = mpl.transforms.blended_transform_factory(axs[i].transData,
                                                             axs[i].transAxes)
            #axs[i].text(std_vcut + 20., vesc_y, 
            #            '$\hat{v}_\mathrm{esc}(v_\mathrm{c})$', 
            #            transform=trans,
            #            fontsize=vesc_fs, rotation=90., color='gray', 
            #            horizontalalignment='left')
        if show_prediction_vcut and std_vcut_type != prediction_vcut_type:
            axs[i].axvline(prediction_vcut, ls='--', c='k', alpha=0.5)

        if show_vcrit:
            # Draw vcrit line
            axs[i].axvline(vcrits[gal], ls='--', c='grey', alpha=0.5)

        axs[i].set_yscale(scale)
        if scale == 'linear':
            order_of_mag = -3
            if gals == 'discs' and i < 8 and i > 3:
                fac = 3.
            else:
                fac = 2.
            axs[i].yaxis.set_major_locator(
                    mpl.ticker.MultipleLocator(
                            base=fac * 10. ** order_of_mag))
            make_sci_y(axs, i, order_of_mag)
        if xtickspace is not None:
            axs[i].xaxis.set_major_locator(
                    mpl.ticker.MultipleLocator(base=xtickspace))
            axs[i].xaxis.set_minor_locator(plt.NullLocator())
        if scale == 'log':
            axs[i].yaxis.set_major_locator(mpl.ticker.LogLocator(
                numticks=999))
            axs[i].yaxis.set_minor_locator(mpl.ticker.LogLocator(
                numticks=999, subs="auto"))

    if xmax is not None or ymin is not None or ymax is not None:
        plt.draw()
        # Need to draw everything before going back and doing this because
        # otherwise we'll lock in the upper y limit (x limit) for each row with
        # the
        # automatically determined upper limit for the galaxy in the first
        # column (row) of that row (column).
        for ax in axs:
            if xmax is not None:
                ax.set_xlim(right=xmax)
            if ymin is not None:
                ax.set_ylim(bottom=ymin)
            if ymax is not None:
                ax.set_ylim(top=ymax)
    if gals == 'discs':
        ax_ylabel = axs[4]
        xlabel_y = 0.05
        legend_y = -0.02
    else:
        ax_ylabel = axs[0]
        xlabel_y = 0.
        legend_y = -0.1
    ax_ylabel.set_ylabel('$g(v_\mathrm{min})'
                         '\,/\,\mathrm{\left[km^{-1}\,s\\right]}$')
    axs[-1].set_xlabel(
        '$v_\mathrm{min}\,/\,\mathrm{\left[km\,s^{-1}\\right]}$')
    # Put the x-axis label where we want it:
    axs[-1].xaxis.set_label_coords(0.5, xlabel_y, transform=fig.transFigure) 
    handles, labels = axs[-1].get_legend_handles_labels()
    if show_std_vcut:
        handles.append(mpl.lines.Line2D(
            [0], [0], color='grey', 
            ls='--', alpha=0.5,
            label=vcut_labels[std_vcut_type]))
    if show_prediction_vcut and prediction_vcut_type != std_vcut_type:
        handles.append(mpl.lines.Line2D(
            [0], [0], color='k', 
            ls='--', alpha=0.5,
            label=vcut_labels[prediction_vcut_type]))
    axs[-1].legend(handles=handles, loc='upper center', 
                   bbox_to_anchor=(.5, legend_y),
                   bbox_transform=fig.transFigure, ncols=legend_ncols)

    if tgt_fname is not None:
        plt.savefig(paths.figures + tgt_fname,
                    bbox_inches='tight',
                    dpi=350)
    plt.show()

    return None 

def plt_halo_integrals_dblscale(gals, df_source,
                                std_vcut_type=None,
                                prediction_vcut_type=None,
                                show_max_hard=True,
                                show_sigmoid_hard=True,
                                show_mao_naive=False,
                                xmax=None,
                                logymin=1.e-5,
                                show_std_vcut=True,
                                tgt_fname=None):
    if show_sigmoid_hard and prediction_vcut_type is None:
        raise ValueError('You must specify the prediction_vcut_type if you'
                         ' show_sigmoid_hard.')
    if (show_max_hard or show_mao_naive) and std_vcut_type is None:
        raise ValueError('You must specify the std_vcut_type if you show_mao'
                         '_naive or show_max_hard.')
    import fitting
    import dm_den
    df = dm_den.load_data(df_source)
    with open('./data/v_pdfs_disc_dz1.0.pkl','rb') as f:
        pdfs=pickle.load(f)
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        params = pickle.load(f)
    if std_vcut_type is not None:
        std_vcut_dict = dm_den.load_vcuts(std_vcut_type, df)
    if prediction_vcut_type is not None:
        prediction_vcut_dict = dm_den.load_vcuts(prediction_vcut_type, df)
    if show_mao_naive:
        fit_mao_naive = fitting.fit_mao_naive_aggp(std_vcut_type, df_source)
    
    Ncols = min(len(gals), 4)
    Nrows = 2
    Ngals = len(gals)
    xfigsize = 4.6 / 2. * Ncols + 1.
    yfigsize = 1.5 * Nrows + 1. 
    xfigsize *= 1.2
    yfigsize *= 1.2
    fig, axs = plt.subplots(Nrows, Ncols, figsize=(xfigsize, yfigsize),
                            sharey='row', sharex=True, dpi=140)
    fig.subplots_adjust(wspace=0.,hspace=0.1)

    vs_hat = np.linspace(0., 820., 300)
    integrals = {}
    for gal in gals:
        vc100 = df.loc[gal, 'vc100']
        integrals[gal] = {}
        # data
        integrals[gal]['gs_data'] = fitting.numeric_halo_integral(
                pdfs[gal]['bins'], 
                pdfs[gal]['ps'])
        # Staudt prediction
        v0 = params['d'] * vc100 ** params['e']
        vdamp = params['h'] * vc100 ** params['j']
        integrals[gal]['gs_staudt'] = fitting.g_smooth_step_max(
                vs_hat, v0, vdamp, params['k'])
        # Staudt prediction with a hard cut
        if show_sigmoid_hard:
            integrals[gal]['gs_staudt_hard'] = fitting. calc_g_general(
                    vs_hat, fitting.pN_max_double_hard,
                    (v0, vdamp, params['k'], prediction_vcut_dict[gal]))
        # Simple maxwellian with v0 = vc
        integrals[gal]['gs_max'] = fitting.g_smooth_step_max(
                vs_hat, vc100 * 100.,
                np.inf, np.inf)
        # Maxwellian with v0 = vc, hard truncation @ vesc
        if show_max_hard:
            integrals[gal]['gs_max_hard'] = fitting.g_smooth_step_max(
                    vs_hat, vc100 * 100.,
                    std_vcut_dict[gal], np.inf)
        # Mao with v0=vc
        if show_mao_naive:
            integrals[gal]['gs_mao_naive'] = fitting.calc_g_general(
                vs_hat,
                fitting.pN_mao,
                (vc100 * 100., std_vcut_dict[gal],
                 fit_mao_naive.params['p'].value))
    def fill_row(i):
        for j, gal in enumerate(gals):
            vs_truth = pdfs[gal]['vs']

            # Plot data
            axs[i, j].stairs(integrals[gal]['gs_data'], pdfs[gal]['bins'], 
                             color='k',
                             label='data', baseline=None)

            # Plot the prediction
            axs[i, j].plot(vs_hat, integrals[gal]['gs_staudt'],
                           label='prediction from $v_\mathrm{c}$',
                           color='C3', zorder=10)

            # Plot the prediction with a hard cut
            if show_sigmoid_hard:
                axs[i, j].plot(vs_hat, integrals[gal]['gs_staudt_hard'],
                               '--', color='C3', zorder=10)
            # Plot simple maxwellian with v0 = vc
            axs[i, j].plot(vs_hat,
                           integrals[gal]['gs_max'],
                           label='std assumption, $v_0=v\mathrm{c}$',
                           color='C0')

            # Plot maxwellian with v0 = vc, hard truncation @ vesc
            if show_max_hard:
                axs[i, j].plot(
                        vs_hat, integrals[gal]['gs_max_hard'],
                        #label='$v_0=v\mathrm{c}$, cut @ $v_\mathrm{esc}$',
                        color='C0', ls='--')

            # Plot the halo integral from using v0=vc with Mao
            if show_mao_naive:
                axs[i, j].plot(vs_hat, integrals[gal]['gs_mao_naive'],
                            label='Mao, $v_0=v_\mathrm{c}$',
                            color='c')
            
            if i == 0:
                loc = [0.97,0.95]
                kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                                  va='top', ha='right',
                                  bbox=dict(facecolor='white', alpha=0.8, 
                                            edgecolor='none'))
                axs[i, j].annotate(gal, loc,
                                   **kwargs_txt)
                axs[i, j].set_yscale('linear')

                order_of_mag = -3
                axs[i, j].ticklabel_format(style='sci', axis='y', 
                                           scilimits=(order_of_mag,
                                                      order_of_mag),
                                           useMathText=True)
            if show_std_vcut:
                # Draw vesc line
                axs[i, j].axvline(std_vcut_dict[gal], ls='--', c='grey', 
                                  alpha=0.5)

            if i == 1:
                axs[i, j].set_yscale('log')
                axs[i, j].yaxis.set_major_locator(mpl.ticker.LogLocator(
                    numticks=999))
                axs[i, j].yaxis.set_minor_locator(mpl.ticker.LogLocator(
                    numticks=999, subs="auto"))

    fill_row(0)
    fill_row(1)

    plt.draw()
    # Need to draw everything before going back and doing this because
    # otherwise we'll lock in the upper y limit (x limit) for each row with
    # the
    # automatically determined upper limit for the galaxy in the first
    # column (row) of that row (column).
    axs[1, 0].set_ylim(bottom=logymin)
    if xmax is not None:
        for j in range(len(gals)):
            axs[1, j].set_xlim(right=xmax)

    axs[0, 0].set_ylabel('$g(v_\mathrm{min})'
                         '\ \mathrm{\left[km^{-1}\,s\\right]}$')
    axs[0, 0].yaxis.set_label_coords(0.03, 0.5, transform=fig.transFigure)
    axs[0, 0].set_xlabel(
            '$v_\mathrm{min}\ \mathrm{\left[km\,s^{-1}\\right]}$')
    axs[0, 0].xaxis.set_label_coords(0.5, 0.03, transform=fig.transFigure)                       

    handles, labels = axs[0, 0].get_legend_handles_labels()
    if show_std_vcut:
        handles.append(mpl.lines.Line2D(
            [0], [0], color='grey', 
            ls='--', alpha=0.5,
            label=vcut_labels[std_vcut_type]))
    axs[0, -1].legend(handles=handles, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.05),
                      bbox_transform=fig.transFigure, ncols=2,
                      handlelength=1.2, columnspacing=0.8)

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=350)

    plt.show()

    return None

def plt_halo_integral_mw(df_source, 
                         tgt_fname=None, ymin=1.e-6, sigmoid_damped_eqnum=None,
                         xtickspace=None,
                         dpi=150):
    import fitting
    import dm_den
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        params = pickle.load(f)
    df = dm_den.load_data(df_source)

    Ncols = 2
    Nrows = 1
    xfigsize = 5. 
    yfigsize = 2. 
    fig, axs = plt.subplots(Nrows, Ncols, figsize=(xfigsize, yfigsize), 
                            dpi=dpi)
    
    vmins = np.linspace(0., 780., 300)
    vc100 = vc_eilers / 100.
    vesc_hat = dm_den.load_vcuts('lim_fit', df)['mw']
    
    gs_max = fitting.g_smooth_step_max(vmins, vc_eilers, np.inf, np.inf)
    gs_max_hard = fitting.g_smooth_step_max(vmins, vc_eilers, vesc_hat, np.inf)

    v0 = params['d'] * vc100 ** params['e']
    vdamp = params['h'] * vc100 ** params['j']
    gs_sigmoid_damped = fitting.g_smooth_step_max(vmins, v0, vdamp, 
                                                  params['k'])
    gs_sigmoid_damped_hard = fitting.calc_g_general(vmins, 
                                                    fitting.pN_max_double_hard,
                                                    args=(v0, vdamp, 
                                                          params['k'], 
                                                          vesc_hat))

    for i in [0, 1]:
        axs[i].plot(vmins, gs_max, label='Maxwellian,\n$v_0=v_{\\rm c}$')
        axs[i].plot(vmins, gs_sigmoid_damped, 
                    label=('$f(v) = {{\\rm Eqn. }}{0:s}$'
                           .format(str(sigmoid_damped_eqnum))),
                    color='C3')
        if xtickspace is not None:
            axs[i].xaxis.set_major_locator(
                    mpl.ticker.MultipleLocator(base=xtickspace))
            axs[i].xaxis.set_minor_locator(plt.NullLocator())
    axs[0].plot(vmins, gs_max_hard, '--', color='C0', 
                label='cut @ $v_{\\rm esc}(v_{\\rm c})$')
    axs[0].plot(vmins, gs_sigmoid_damped_hard, '--', color='C3',
                label='cut @ $v_{\\rm esc}(v_{\\rm c})$')

    axs[1].yaxis.set_major_locator(
            mpl.ticker.MultipleLocator(base=1.e-3))
    axs[1].xaxis.set_minor_locator(plt.NullLocator())
    make_sci_y(axs, 1, -3)
    axs[1].set_xlim(right=670.)

    axs[1].set_xlabel('$v_\mathrm{min}\,/\,\mathrm{\left[km\,s^{-1}\\right]}$')
    # Put the x-axis label where we want it:
    axs[1].xaxis.set_label_coords(0.5, 0., transform=fig.transFigure)                       

    axs[0].set_ylabel('$g(v_{\\rm min})\,/\,\\rm\left[km^{-1}\,s\\right]$')

    axs[0].set_yscale('log')
    axs[0].set_ylim(bottom=ymin)
    # Make minor log ticks
    axs[0].yaxis.set_major_locator(mpl.ticker.LogLocator(
        numticks=999))
    axs[0].yaxis.set_minor_locator(mpl.ticker.LogLocator(
        numticks=999, subs="auto"))

    axs[1].legend(bbox_to_anchor=(1., 0.), borderaxespad=0.)
    handles, labels = axs[0].get_legend_handles_labels()
    trans = mpl.transforms.blended_transform_factory(axs[1].transAxes,
                                                     fig.transFigure)
    axs[1].legend(handles=[handles[i] for i in [0, 2, 1, 3]],
                  bbox_to_anchor=(1., 0.5), 
                  loc='center left', ncol=1,
                  bbox_transform=trans,
                  borderaxespad=1.)

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=350)
    plt.show()

    return None

def plt_halo_integral_mw_with_ratio(df_source, 
                         tgt_fname=None, sigmoid_damped_eqnum=None,
                         xtickspace=None,
                         dpi=150):
    import fitting
    import dm_den
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        params = pickle.load(f)
    df = dm_den.load_data(df_source)

    Ncols = 1
    Nrows = 2
    xfigsize = 5. 
    yfigsize = 3.5 
    fig, axs = plt.subplots(Nrows, Ncols, figsize=(xfigsize, yfigsize), 
                            dpi=dpi, sharex=True)
    fig.subplots_adjust(hspace=0.)
    
    vmins = np.linspace(0., 780., 300)
    vc100 = vc_eilers / 100.
    vesc_hat = dm_den.load_vcuts('lim_fit', df)['mw']
    
    gs_max = fitting.g_smooth_step_max(vmins, vc_eilers, np.inf, np.inf)
    gs_max_hard = fitting.g_smooth_step_max(vmins, vc_eilers, vesc_hat, np.inf)

    v0 = params['d'] * vc100 ** params['e']
    vdamp = params['h'] * vc100 ** params['j']
    gs_sigmoid_damped = fitting.g_smooth_step_max(vmins, v0, vdamp, 
                                                  params['k'])
    gs_sigmoid_damped_hard = fitting.calc_g_general(vmins, 
                                                    fitting.pN_max_double_hard,
                                                    args=(v0, vdamp, 
                                                          params['k'], 
                                                          vesc_hat))

    axi_ratio = 1
    axi_linear = 0

    axs[axi_ratio].plot(
            vmins, (gs_sigmoid_damped / gs_max),
            color='C3',
            label='Eq. {0:s} / Maxwellian'.format(str(sigmoid_damped_eqnum))
    )
    axs[axi_ratio].plot(
            vmins, (gs_sigmoid_damped_hard / gs_max_hard),
            color='C3', ls='--',
            label='cut Eq. {0:s} / cut Maxwellian'\
                  .format(str(sigmoid_damped_eqnum))
    )
    axs[axi_ratio].axhline(1., lw=1., ls='--', color='grey')
                    
    axs[axi_linear].plot(vmins, gs_max, label='Maxwellian,\n$v_0=v_{\\rm c}$')
    axs[axi_linear].plot(vmins, gs_sigmoid_damped, 
                label=('$f(v) = {{\\rm Eqn. }}{0:s}$'
                       .format(str(sigmoid_damped_eqnum))),
                color='C3')

    for i in [0, 1]:
        if xtickspace is not None:
            axs[i].xaxis.set_major_locator(
                    mpl.ticker.MultipleLocator(base=xtickspace))
            axs[i].xaxis.set_minor_locator(plt.NullLocator())
    del i

    axs[axi_linear].yaxis.set_major_locator(
            mpl.ticker.MultipleLocator(base=1.e-3))
    axs[axi_linear].xaxis.set_minor_locator(plt.NullLocator())
    axs[axi_linear].set_xlim(right=670.)

    axs[axi_linear].set_xlabel(
            '$v_\mathrm{min}\,/\,\mathrm{\left[km\,s^{-1}\\right]}$'
    )
    # Put the x-axis label where we want it:
    axs[axi_linear].xaxis.set_label_coords(0.5, 0., transform=fig.transFigure)

    # Put linear plot in scientific notation
    order_of_mag = -3
    axs[axi_linear].yaxis.set_major_formatter(
            lambda y, pos: '{0:0.0f}'.format(y / 10.**order_of_mag))
    
    axs[axi_linear].set_ylabel(
            '$\dfrac{{g(v_{{\\rm min}})}}'
            '{{10^{{{0:0.0f}}}\\rm\,km^{{-1}}\,s}}$'
            .format(order_of_mag)
    )
    # Put the linear y-axis label where we want it:
    axs[axi_linear].yaxis.set_label_coords(-0.1, 0.5)

    axs[axi_ratio].set_ylabel(
            '$\dfrac{{g_{{{0:s}}}}}{{g_\mathrm{{Maxwellian}}}}$'
            .format(str(sigmoid_damped_eqnum))
    )

    axs[axi_linear].set_ylim(bottom=-0.5e-3)

    # Set y-tick spacing for the ratio panel
    axs[axi_ratio].yaxis.set_major_locator(
            mpl.ticker.MultipleLocator(base=0.2)
    )

    handles, labels = axs[axi_ratio].get_legend_handles_labels()
    trans = mpl.transforms.blended_transform_factory(axs[axi_linear].transAxes,
                                                     fig.transFigure)
    axs[axi_linear].legend(
            #handles=[handles[i] for i in [0, 2, 1, 3]],
            bbox_to_anchor=(1., 0.5), 
            loc='center left', ncol=1,
            bbox_transform=trans,
            borderaxespad=1.
    )

    axs[0].tick_params(axis='x', direction='inout', length=6)

    loc = [0.97,0.96]
    kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                      va='top', ha='right',
                      bbox=dict(facecolor='white', alpha=0.8, 
                                edgecolor='none', pad=0.1))
    axs[0].annotate('Milky Way', loc,
                    **kwargs_txt)

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=350)
    plt.show()

    return None

def plt_anisotropy(df_source, only_discs=True, savefig=False, vertical=False,
                   xticklabel_fontsize=10, figsize=None):
    df_copy = dm_den.load_data(df_source)
    if only_discs:
        df_copy.drop(['m12z', 'm12w'], inplace=True)
    df_copy['$\sigma_z/\sigma_r$'] = df_copy['std(v_dot_zhat_disc(dm))']\
                                   /df_copy['std(v_dot_rhat_disc(dm))']
    df_copy['$\sigma_\phi/\sigma_r$'] = df_copy['std(v_dot_phihat_disc(dm))']\
                                      /df_copy['std(v_dot_rhat_disc(dm))']
    df_copy['$\sigma_z/\sigma_\phi$'] = df_copy['std(v_dot_zhat_disc(dm))'] \
                                      / df_copy['std(v_dot_phihat_disc(dm))']
    df_copy['$\\beta$'] = 1. - (df_copy['std(v_dot_phihat_disc(dm))']**2. \
                              + df_copy['std(v_dot_zhat_disc(dm))']**2.) \
                             / (2.*df_copy['std(v_dot_rhat_disc(dm))']**2.)
    df_copy['$\\beta_\phi$'] = 1. - df_copy['std(v_dot_phihat_disc(dm))']**2. \
                             / df_copy['std(v_dot_rhat_disc(dm))']**2.
    df_copy['$\\beta_z$'] = 1. - df_copy['std(v_dot_zhat_disc(dm))']**2. \
                             / df_copy['std(v_dot_rhat_disc(dm))']**2.
    display(df_copy)
    print('fraction(beta < 0.25) = {0:0.4f}'
          .format((df_copy['$\\beta$'] < 0.25).sum() / len(df_copy)))

    if vertical:
        if figsize is None:
            figsize = (4.5, 8.)
        position1 = 212
        position2 = 211
    else:
        if figsize is None:
            figsize = (8.5, 2.5)
        position1 = 122
        position2 = 121
        
    fig = plt.figure(figsize=figsize, dpi=130)
    ax1 = fig.add_subplot(position1)
    fig.subplots_adjust(wspace=0.15, hspace=0.3)

    df_copy[['$\sigma_\phi/\sigma_r$', 
             '$\sigma_z/\sigma_r$',
             '$\sigma_z/\sigma_\phi$']].plot.bar(ax=ax1, 
                                                 color=['#17becf',
                                                        '#ff7f0e',
                                                        '#9467bd'],
                                                 width=0.6)
    ax2 = fig.add_subplot(position2)
    df_copy[['$\\beta$', 
             '$\\beta_\phi$',
             '$\\beta_z$']].plot.bar(ax=ax2, color=['k','#17becf','#ff7f0e'])
    ax2.axhline(0., color='grey', lw=0.5, ls=(0, (5, 6)))

    if vertical:
        legend_kwargs = dict(
            bbox_to_anchor=(0., 1.), loc='upper left', 
            ncol=3, borderpad=0.2, handlelength=1.3,
            columnspacing=1., handletextpad=0.5,
            borderaxespad=0.4
        )
        
        ax1.set_ylim(0.75, 1.12)
        ax2.set_ylim(None, 0.48)
        ax1.legend(**legend_kwargs)
        ax2.legend(**legend_kwargs)
        #ax2.set_ylim(0.75,None)
    else:
        legend_y = -0.25
        # ax1 is actually on the right.
        ax1.legend(bbox_to_anchor=(0.5,legend_y), loc="upper center", ncol=3,
                   columnspacing=0.7)
        ax2.legend(bbox_to_anchor=(0.5,legend_y), loc="upper center", ncol=3)
        #ax2.set_ylim(0.75,None)
    ax1.set_axisbelow(True)
    ax1.xaxis.grid(False)
    ax2.set_axisbelow(True)
    ax2.xaxis.grid(False)

    for ax in [ax1,ax2]:
        labels = ax1.xaxis.get_majorticklabels()
        ax.set_xticklabels(labels, rotation=30, ha='right', 
                           rotation_mode='anchor')
        ax.tick_params('x', labelsize=xticklabel_fontsize)

    if savefig:
        plt.savefig(paths.figures+'anisotropy.png',
                    bbox_inches='tight', dpi=350)
    plt.show()

def setup_multigal_fig(gals, show_resids=True, sharey='row'):
    islist = isinstance(gals, (list, np.ndarray, 
                               pd.core.indexes.base.Index))
    if islist:
        Ncols = min(len(gals), 4)
        Nrows = math.ceil(len(gals) / Ncols)
        Ngals = len(gals)
    elif gals == 'discs':
        figsize = (19., 12.)
        Nrows = 3
        Ncols = 4
        Ngals = 12
    xfigsize = 4.6 / 2. * Ncols + 1.
    yfigsize = 1.5 * Nrows + 1.
    if Ngals <= 4 and show_resids:
        # Add room for residual plots
        Nrows += 1
        yfigsize += 1. 
        height_ratios = [4,1]
    else:
        height_ratios = None
    fig,axs=plt.subplots(Nrows, Ncols, figsize=(xfigsize, yfigsize), 
                         sharey=sharey,
                         sharex=True, dpi=140, height_ratios=height_ratios)
    if Ngals == 1:
        axs=[axs]
    else:
        axs=axs.ravel()
    fig.subplots_adjust(wspace=0.,hspace=0.)
    fig.Nrows = Nrows
    fig.Ncols = Ncols

    return fig, axs

def label_axes(axs, fig, gals):
    if not isinstance(gals, (list, np.ndarray, pd.core.indexes.base.Index)) \
       and gals == 'discs':
        for i in [4]:
            axs[i].set_ylabel('$f(v)\,4\pi v^2\,/\,[\mathrm{km^{-1}\,s}]$')
        xlabel_yloc = 0.05
    elif len(gals) < 4:
        axs[0].set_ylabel('$f(v)\,4\pi v^2\,/\,[\mathrm{km^{-1}\,s}]$')
        xlabel_yloc = 0.02
    else:
        xlabel_yloc = 0.04
    axs[0].set_xlabel('$v\,/\,[\mathrm{km\,s^{-1}}]$')
    axs[0].xaxis.set_label_coords(0.5, xlabel_yloc, transform=fig.transFigure)
    return None

if __name__=='__main__':
    fname=sys.argv[1]
    df=dm_den.load_data(fname)
    #Take each dict of inputs from make_plot_feed() and make a plot with each:
    for f in make_plot_feed(df):
        plotter(**f)
