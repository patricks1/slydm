from scipy.stats.stats import pearsonr
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
import lmfit
import staudt_fire_utils as utils
from progressbar import ProgressBar

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
rcParams['legend.fontsize']=18
rcParams['figure.facecolor'] = (1., 1., 1., 1.) #white with alpha=1.

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

disp_vir_label='$\sigma(R_\mathrm{vir})\ [\,\mathrm{km}\cdot\mathrm{s}^{-1}\,]$'
disp_solar_label='$\sigma(R_0)\ [\,\mathrm{km}\cdot\mathrm{s}^{-1}\,]$'
disp_dm_solar_label='$\sigma_\mathrm{DM}(R_0)\ [\,\mathrm{km}\cdot\mathrm{s}^{-1}\,]$'
disp_gas_solar_label='$\sigma_\mathrm{gas}(R_0)\ [\,\mathrm{km}\cdot\mathrm{s}^{-1}\,]$'
log_disp_solar_label='$\log(\,\sigma(R_0)\,/\,[\,\mathrm{km}\cdot\mathrm{s}^{-1}\,]\,)$'
m_label='$M_\mathrm{vir}\ [\mathrm{M}_\odot]$'
mbtw_label='$M(10\,\mathrm{kpc}<r<R_\mathrm{vir})\[\mathrm{M}_\odot]$'
log_rho_solar_label='$\log(\,\\rho(R_0)\,/\,[\,\mathrm{M}_\odot'\
                    '\mathrm{kpc}^{-3}\,]\,)$'
log_rho_label='$\log(\,\\rho(R_\mathrm{vir})\,/\,[\,\mathrm{M}_\odot\mathrm{kpc}^{-3}\,]\,)$'
rho_label='$\\rho(R_\mathrm{vir})\;[\,\mathrm{M}_\odot\mathrm{kpc}^{-3}\,]$'
den_label = '$\\rho\,/\,\\left[\mathrm{M_\odot kpc^{-3}}\\right]$'
disp_label = '$\\sigma_\mathrm{3D}\,/\,'\
             '\\left[\mathrm{km\,s^{-1}}\\right]$'
gmr_label = '$\sqrt{Gm/R_0}\,/\,'\
              '\\left[\mathrm{km\,s^{-1}}\\right]$'
vc_label = '$v_\mathrm{c}\,/\,[\mathrm{km\,s^{-1}}]$'

# vc from Eilers et al. 2019
vc_eilers = 229.
dvc_eilers = 6.

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

def ax_slr(ax, fname, xcol, ycol,
           xlabel,ylabel,
           xadjustment=None, yadjustment=None,
           xscale='linear', yscale='linear', 
           showlabels=True,
           labelsize=15, arrowprops=None, formula_y=-0.2,
           dropgals=None, showGeV=True, show_formula=True,
           x_forecast=None, dX=None, forecast_sig=1.-0.682, verbose=False,
           adjust_text_kwargs={}, legend_txt=None, 
           return_error=False, show_band=False,
           **kwargs):
    'Plot a simple linear regression on ax'

    def plt_forecast(ax, X_forecast, Yhat, dYhat):
        if xadjustment=='log':
            X_forecast = np.log10(X_forecast)
        else:
            X_forecast = np.array(X_forecast)
        # Returns a list of errorbar objects 
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
                yerr = np.array([[dYhat[i][1]], [dYhat[i][0]]])
            elif len(dYhat[i]) == 1:
                # symmetric error
                yerr = dYhat[i][0]
            else:
                raise ValueError('dYhat[{0:0.0f}] has an unexpected shape.'
                                 .format(i))
            eb_add = ax.errorbar(X_forecast.flatten()[i], 
                         Yhat[i,0],
                         yerr=yerr,
                         c=color, capsize=3,
                         marker='o', ms=8, 
                         mec=color, mfc=color
                         )
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
        raise ValueError('x adjustment should be \'log\' or None')
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
    coefs, intercept, r2, Xs, ys, ys_pred, r2a, resids, delta_beta, band \
            = mlr_res[:10]

    if show_band:
        band_disp = band.copy()
        if xadjustment == 'logreg_linaxunits':
            band_disp[0] = 10.**band_disp[0]
        if yadjustment == 'logreg_linaxunits':
            band_disp[1:] = 10.**band_disp[1:]

        ax.fill_between(*band_disp, color='grey', alpha=0.2, lw=0.)

    if x_forecast is not None:
        prediction_y = mlr_res[-1] #[y, y uncertainty] 
        # ebc is an ErrorbarContainer. I think by telling adjust_texts to avoid
        # ebc[0], it will avoid the prediction point.
        if yadjustment == 'logreg_linaxunits':
            prediction_y = np.array(prediction_y, dtype=object)
            #if prediction_y.shape[1:] != (1,1):
            Ys = np.zeros(prediction_y.shape[1:])
            dYs = []
            for i, (Y, dY) in enumerate(zip(prediction_y[0], 
                                          prediction_y[1])):
                conversion = staudt_utils.log2linear(Y[0], dY[0])
                Ys[i,0] = conversion[0]
                dYs.append(conversion[1:])
        else:
            Ys = prediction_y[0]
            dYs = prediction_y[1]
        eb = plt_forecast(ax, x_forecast, Ys, dYs)
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
        ys_pred = 10.**ys_pred
    if xadjustment == 'logreg_linaxunits':
        Xs[0] = 10.**Xs[0]
    ax.plot(Xs[0], ys_pred, label=legend_txt) #Plot the regression line

    if show_formula:
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
                ax.annotate('${0:s}={1:0.2f}{2:s}{5:s}{3:0.2f}$\n'
                            '$r^2={4:0.2f}$'\
                            .format(ystring, coefs[0], xstring, 
                                    np.abs(intercept), r2,
                                    operator),
                            (0., formula_y),
                            xycoords='axes fraction', fontsize=18)

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
    ax2.set_xlabel('$\mathrm{GeV}\,c^{-2}\,\mathrm{cm^{-3}}$', 
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

    ax2.set_ylim(x0,x1)
    ax2.set_yticks(visible_ticks)
    ax2.set_yticklabels(['{:,.2f}'.format(lab.value) for lab in labs], 
                        fontsize=12.)
    ax2.set_ylabel('$\mathrm{GeV}\,c^{-2}\,\mathrm{cm^{-3}}$', 
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
        c = np.log10(c)
        cmap = cmr.get_sub_cmap('viridis', 0., 0.9)
        sc = ax.scatter(xs,ys,marker='o',c=c,alpha=alpha,label=legend_txt,
                        cmap=cmap)
        '''
        if ycol == 'den_disc':
            pad = 0.25
        else:
            pad = 0.05
        '''
        cb = plt.colorbar(sc, pad=0.2, 
                          location='bottom')
        cb.ax.tick_params(labelsize=12)
        cb.set_label(size=12,
                     label='$\log M_\mathrm{\star,vir}\,/\,\mathrm{M_\odot}$')
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
            adjust_text(texts, ax=ax)

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

def plt_vs_vc(ycol, tgt_fname, source_fname='dm_stats_dz1.0_20230626.h5',
              update_val=False,
              forecast_sig=1.-0.682, #forecast significance
              verbose=False, 
              adjust_text_kwargs={}, show_formula='outside',
              figsize=(10,5), labelsize=14., vc=vc_eilers, dvc=dvc_eilers,
              label_overrides={},
              **kwargs):
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
    if len(label_overrides) > 0:
        for gal in label_overrides:
            if len(label_overrides[gal]) != 3:
                raise ValueError('`label_overrides` must be in the form '
                                 '{galname: (annx, anny, draw_arrow)}.')
    vc /= 100.
    dvc /= 100.

    df = dm_den.load_data(source_fname)
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
                      #'v_dot_phihat_disc(T<=1e4)',
                      'vc100',
                      ycol,
                      xlabel=vc_label,
                      ylabel=ylabel,
                      xadjustment='logreg_linaxunits', yadjustment=yadjustment,
                      xscale='log', yscale=yscale,
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

    if update_val:
        y_flat = np.array(yhat_vc, dtype=object).flatten()

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
            y_save, dy_save = staudt_utils.sig_figs(*y_flat)
            dm_den.save_prediction('disp', y_save, dy_save)
            dm_den.save_prediction('disp_slope', slope, dslope_str)
            disp_transform = staudt_utils.log2linear(logy_intercept_raw,
                                                     delta_beta[0][0])
            disp_amp = disp_transform[0]
            ddisp_amp = disp_transform[1:] 
            disp_amp_str, ddisp_amp_str = staudt_utils.sig_figs(disp_amp, 
                                                                ddisp_amp)
            dm_den.save_prediction('disp_amp', disp_amp_str, ddisp_amp_str)
        elif ycol=='den_disc':
            data2save = {'den_slope': slope_raw, 
                         'logden_intercept': logy_intercept_raw}
            dm_den.save_var_raw(data2save) 

            y_save, dy_save = staudt_utils.sig_figs(*y_flat)
            dm_den.save_prediction('logrho', y_save, dy_save)

            # I expect log2linear to return asymetric errors in the following 
            # line.
            Y_MSUN = staudt_utils.log2linear(*y_flat) * u.M_sun/u.kpc**3.
            Y_1E7MSUN = Y_MSUN / 1.e7 
            y_1e7msun_txt, DY_1E7MSUN_TXT = staudt_utils.sig_figs(
                    Y_1E7MSUN[0].value, Y_1E7MSUN[1:].value) 
            dm_den.save_prediction('rho_1e7msun', y_1e7msun_txt, 
                                   DY_1E7MSUN_TXT)

            particle_y = (Y_MSUN * c.c**2.).to(u.GeV/u.cm**3.) 
            # The following variable assignment is written in such a way that 
            # it will work 
            # no matter whether
            # the errors are symmetric or asymetric (with the [1:])
            particle_y_save, particle_dy_save = staudt_utils.sig_figs(
                   particle_y[0].value,
                   particle_y[1:].value)
            dm_den.save_prediction('rho_GeV', particle_y_save, 
                                   particle_dy_save)
            
            dm_den.save_prediction('den_slope', slope, dslope_str)
            dm_den.save_prediction('logden_intercept', logy_intercept_str, 
                                   dlogy_intercept_str)
            RHO0_1E7MSUN = staudt_utils.log2linear(
                    logy_intercept_raw, dlogy_intercept_raw) / 1.e7 
            RHO0_1E7MSUN *= u.M_sun / u.kpc**3.
            RHO0_GEV = (RHO0_1E7MSUN * 1.e7 * c.c**2.).to(u.GeV / u.cm**3.) 
            rho0_1e7msun_txt, DRHO0_1E7MSUN_TXT = staudt_utils.sig_figs(
                    RHO0_1E7MSUN[0].value, RHO0_1E7MSUN[1:].value)
            rho0_GeV_txt, DRHO0_GEV_TXT = staudt_utils.sig_figs(
                    RHO0_GEV[0].value, RHO0_GEV[1:].value)
            dm_den.save_prediction('rho0_1e7msun', rho0_1e7msun_txt,
                                   DRHO0_1E7MSUN_TXT)
            dm_den.save_prediction('rho0_GeV', rho0_GeV_txt, DRHO0_GEV_TXT)

    display(Latex('$r=8.3\pm{0:0.2f}\,\mathrm{{kpc}}$'
                  .format(df.attrs['dr']/2., df.attrs['dz']/2.)))
    display(Latex('$|z|\in[0,{1:0.2f}]\,\mathrm{{kpc}}$' \
                  .format(df.attrs['dr']/2., df.attrs['dz']/2.)))

    # Take the vc/100 values that the x-axis is based on and turn them into vc:
    ax.xaxis.set_major_formatter(lambda x, pos: '{0:0.0f}'.format(x*100.))

    # Replace the necessary labels/annotations with their overrides
    for child in ax.get_children():
        if isinstance(child, mpl.text.Annotation):
            text = child.get_text()
            if text in label_overrides:
                child.remove() # Delete the existing text label

                # Get the location of the data point
                data_y = df.loc[text, ycol]
                if ycol == 'den_disc':
                    data_y = np.log10(data_y)
                data_x = df.loc[text, 'vc100']
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
    plt.draw()
    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=140)

    plt.show()
    
    return yhat_vc

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
            if update_val:
                raise ValueError('We only save our result in log.')
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
            print('{0:0.2f}% max den diff'.format(den_percent_diff))
            print('{0:0.2f}% max disp diff'.format(disp_percent_diff))
            if update_val:
                #update the value in data.dat for the paper
                dm_den.save_var_latex('maxdiff',
                                      '{0:0.2f}\%'.format(percent_diff))
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
    axs[-2].xaxis.set_major_formatter(lambda x, pos: '{0:0.3f}'.format(x))
    axs[-1].xaxis.set_major_formatter(lambda x, pos: '{0:0.3f}'.format(x))

    if tgt_fname:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=140)

    plt.show()
    
    return None

def plt_gmr_vs_vc(df_source='dm_stats_dz1.0_20230626.h5', tgt_fname=None,
                  figsize=(8,4),
                  labelsize=11., adjust_text_kwargs={}):
    df = dm_den.load_data(df_source).drop(['m12z','m12w'])
    xcol = 'v_dot_phihat_disc(T<=1e4)'
    ycol = 'vcirc'

    fig = plt.figure(figsize=figsize, dpi=110)
    ax = fig.add_subplot(111)
    ax_slr(ax, df_source, 
           xcol, ycol, 
           xlabel=vc_label, ylabel=gmr_label, 
           xadjustment='log',
           yadjustment='log',
           show_formula='outside', dropgals=['m12z','m12w'],
           labelsize=labelsize, arrowprops={'arrowstyle':'-'},
           legend_txt='best fit',
           adjust_text_kwargs=adjust_text_kwargs)
    
    #Plot 1:1 line
    xs = np.log10(df[xcol])
    ys = np.log10(df[ycol])
    ax.plot([xs.min(), xs.max()], [xs.min(), xs.max()], color='gray', 
            ls='--', label='1:1')
    errors = (ys-xs).values
    sse = errors.T @ errors
    diffs = ys-np.mean(ys)
    tss = diffs.T @ diffs
    r2_1to1 = 1.-sse/tss
    display(Latex('$r^2_\mathrm{{1:1}}={0:0.2f}$'.format(r2_1to1)))

    ax.legend(fontsize=11)

    if tgt_fname:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=140)

    plt.show()

    return None

def plt_particle_counts(df_source):
    import cropper

    df = dm_den.load_data(df_source)
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

    return None

def plt_universal_prefit(result, gals='discs', ddfrac=None, dhfrac=None, 
                         ymax=None):
    import dm_den
    import fitting
    if gals != 'discs' and not isinstance(gals, (list, np.ndarray)):
        raise ValueError('Unexpected value provided for gals arg')
    if ddfrac is None or dhfrac is None:
        grid_results = grid_eval.identify()
        if ddfrac is None:
            ddfrac = grid_results[0]
            print('Using ddfrac = {0:0.5f}'.format(ddfrac))
        if dhfrac is None:
            dhfrac = grid_results[1]
            print('Using dhfrac = {0:0.5f}'.format(dhfrac))
    df = dm_den.load_data('dm_stats_20221208.h5').drop(['m12w', 'm12z'])
    with open('./data/v_pdfs.pkl','rb') as f:
        pdfs=pickle.load(f)
    pdfs.pop('m12z')
    pdfs.pop('m12w')
    if gals == 'discs':
        gals = pdfs.keys() #Change `gals` from a str to a list-like of galnames
    Ngals = len(gals) 
    N_postfit = 300
    vs_postfit = np.linspace(0., 700., N_postfit)
    
    fig, axs = fitting.setup_universal_fig(gals)
    if type(result) == lmfit.model.ModelResult:
        d, e, h, j, k = [result.params[key] 
                         for key in ['d', 'e', 'h', 'j', 'k']]
    elif type(result) == dict:
        d, e, h, j, k = [result[key]
                         for key in ['d', 'e', 'h', 'j', 'k']]
    else:
        raise ValueError('Unexpected data type provided for `params`')
    for i, gal in enumerate(df.index):
        vc = df.loc[gal, 'v_dot_phihat_disc(T<=1e4)']
        v0 = d * (vc / 100.) ** e
        vdamp = h * (vc / 100.) ** j
        ps_postfit = fitting.smooth_step_max(vs_postfit,
                                             v0, vdamp,
                                             k)

	# Error bands
        samples_color = plt.cm.viridis(0.5)
        lowers, uppers = fitting.gal_bands(gal, vs_postfit, df, 
				           result, ddfrac=ddfrac, 
                                           dhfrac=dhfrac,
                                           assume_corr=False,
                                           ax = axs[i], 
                                           samples_color=samples_color)
        axs[i].fill_between(vs_postfit, lowers, uppers, 
                            color=plt.cm.viridis(1.), 
			    alpha=0.9, 
			    ec=samples_color, zorder=1, 
			    label='$1\sigma$ band')

        # Plot data
        axs[i].stairs(pdfs[gal]['ps'], pdfs[gal]['bins'], color='k',
                      label='data')
        # Plot prediction
        axs[i].plot(vs_postfit,
                    ps_postfit,
                    '-',
                    label='prediction from $v_\mathrm{c}$', color='C3', lw=1.5)
        # Make ticks on both sides of the x-axis:
        axs[i].tick_params(axis='x', direction='inout', length=6)
        # Put y-axis in scientific notation
        order_of_mag = -3
        axs[i].ticklabel_format(style='sci', axis='y', 
                                scilimits=(order_of_mag,
                                           order_of_mag),
                                useMathText=True)
        if Ngals == 2:
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
            axs[i+2].plot(vs_resids, resids / 10.**order_of_mag, color='C3')
            axs[i+2].axhline(0., linestyle='--', color='k', alpha=0.5, lw=1.)
            axs[i+2].set_ylim(-resids_lim, resids_lim)
            if i == 0:
                axs[i+2].set_ylabel('resids')
        loc = [0.97,0.96]
        if Ngals == 12:
            namefs = 13. #Font size for galaxy name
            vcfs = 9. #Font size for circular velocity
        else:
            namefs = 16. #Font size for galaxy name
            vcfs = 11. #Font size for circular velocity
        kwargs_txt = dict(fontsize=namefs, xycoords='axes fraction',
                          va='top', ha='right',
                          bbox=dict(facecolor='white', alpha=0.8, 
                                    edgecolor='none'))
        axs[i].annotate(gal, loc,
                        **kwargs_txt)
        loc[1] -= 0.15
        kwargs_txt['fontsize'] = vcfs 
        #rms_err = fitting.calc_rms_err(pdfs[gal]['vs'], pdfs[gal]['ps'],
        #                               fitting.smooth_step_max,
        #                               (v0, vdamp, k))
        #rms_txt = staudt_utils.mprint(rms_err, d=1, show=False).replace('$','')
        axs[i].annotate('$v_\mathrm{{c}}={0:0.0f}\,\mathrm{{km\,s^{{-1}}}}$'
                        #'\n$\mathrm{{RMS}}_\mathrm{{err}}={1:s}$'
                        .format(vc, 
                                #rms_txt
                               ),
                        loc, **kwargs_txt)
        axs[i].grid(False)
        if ymax is not None:
            axs[i].set_ylim(top=ymax)
    if fig.Nrows == 3:
        legend_y = 0.
        ncol = 4
    else:
        legend_y = -0.04
        ncol = 2
    fig.suptitle('Fit based on velocity distrib', fontsize=18.)
    return None

def plt_mw(tgt_fname=None, dvc=0.):
    import grid_eval
    import dm_den
    import fitting
    with open(paths.data + 'data_raw.pkl', 'rb') as f:
        results = pickle.load(f)
    ddfrac, dhfrac = grid_eval.identify()

    vc = vc_eilers
    vs = np.linspace(0., 750., 300)

    def predict(vc, ax, **kwargs):
        df = dm_den.load_data('dm_stats_20221208.h5')
        df.loc['mw', 'v_dot_phihat_disc(T<=1e4)'] = vc
        v0 = results['d'] * (vc / 100.) ** results['e']
        vdamp = results['h'] * (vc / 100.) ** results['j']
        ps = fitting.smooth_step_max(vs, v0, vdamp, results['k'])
        ax.plot(vs, ps, label='prediction from $v_\mathrm{c}$',
                #label = '$v_\mathrm{{c}} = {0:0.0f}\,\mathrm{{km\,s^{{-1}}}}$'\
                #        .format(vc),
                **kwargs)
        lowers, uppers = fitting.gal_bands('mw', vs, df, results, ddfrac, 
                                           dhfrac, 
                                           ax=None, dvc=dvc)
        ax.fill_between(vs, lowers, uppers, 
                        alpha=0.9, 
                        color='#c0c0c0',
                        zorder=1, 
                        label='$1\sigma$ band')
        return None
    
    fig = plt.figure(figsize = (4.6 / 2. + 1., 2.5), dpi=600,
                     facecolor = (1., 1., 1., 0.))
    #fig = plt.figure(figsize = (5., 2.5), dpi=200)
    ax = fig.add_subplot(111)

    #predict(228., ax, c='C0', lw=3., dashes=[2., 0.5])
    predict(vc, ax, c='C3')

    ax.set_ylabel('$f(v)\,4\pi v^2\ [\mathrm{km^{-1}\,s}]$')
    ax.set_xlabel('$v\ [\mathrm{km\,s^{-1}}]$')
    ax.set_ylim(0., None)
    loc = [0.97,0.96]
    kwargs_txt = dict(fontsize=16., xycoords='axes fraction',
                      va='top', ha='right',
                      bbox=dict(facecolor='white', alpha=0.8, 
                                edgecolor='none'))
    ax.annotate('Milky Way', loc,
                **kwargs_txt)
    loc[1] -= 0.15
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
    ax.legend(bbox_to_anchor=(0.5, -0.1), 
              loc='upper center', ncol=1,
              bbox_transform=fig.transFigure)
    #ax.legend(bbox_to_anchor=(0., -0.09), 
    #          loc='upper left', ncol=2,
    #          bbox_transform=fig.transFigure)

    if tgt_fname is not None:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=250)

    plt.show()

if __name__=='__main__':
    fname=sys.argv[1]
    df=dm_den.load_data(fname)
    #Take each dict of inputs from make_plot_feed() and make a plot with each:
    for f in make_plot_feed(df):
        plotter(**f)
