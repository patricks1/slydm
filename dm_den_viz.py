from scipy.stats.stats import pearsonr
from adjustText import adjust_text
from IPython.display import display, Latex
import numpy as np
import pandas as pd
import dm_den
import sys
import paths
import staudt_utils
import pickle
import itertools
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
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['axes.grid']=True
rcParams['axes.titlesize']=24
rcParams['axes.labelsize']=20
rcParams['axes.titlepad']=15
rcParams['legend.frameon'] = True
rcParams['legend.facecolor']='white'
rcParams['legend.fontsize']=18
rcParams['figure.facecolor'] = 'white'

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
disp_label = '$\\sigma_\mathrm{DM}\,/\,'\
             '\\left[\mathrm{km\,s^{-1}}\\right]$'
gmr_label = '$\sqrt{Gm/R_0}\,/\,'\
              '\\left[\mathrm{km\,s^{-1}}\\right]$'
vc_label = '$v_\mathrm{c}\,/\,[\mathrm{km\,s^{-1}}]$'

# v0 from Eilers et al. 2019
v0_eilers = 229.
dv0_eilers = 0.2

# v0 ranges from Sofue 2020
v0_sofue=238.
dv0_sofue=14.
log_dv0_neg = np.log10(v0_sofue/(v0_sofue-dv0_sofue))
log_dv0_pos = np.log10((v0_sofue+dv0_sofue)/v0_sofue)

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
            tgt_fname=None, minarrow=0.02, ax_slr_kwargs={}):
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
            adjust_text_kwargs=adjust_text_kwargs, minarrow=minarrow,
            **ax_slr_kwargs)

    if tgt_fname is not None:
        plt.draw()
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=140)
    plt.show()

    if 'prediction_x' in ax_slr_kwargs:
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
           prediction_x=None, dX=None, fore_sig=1.-0.682, verbose=False,
           minarrow=0.02, adjust_text_kwargs={}, legend_txt=None, **kwargs):
    'Plot a simple linear regression on ax'

    def plt_forecast(ax, x_forecast, yhat):
        delta_f = yhat[1] #uncertainty in the forecast
        if xadjustment=='log':
            x_forecast = np.log10(x_forecast)
        # Returns a list of errorbar objects 
        N = len(x_forecast)
        eb = []
        colors = [plt.cm.cool(i) 
                  for i in np.linspace(0, 
                                       1, 
                                       N)]
        colors = itertools.cycle(['r']+colors)
        for i in range(N):
            color = next(colors)
            eb_add = ax.errorbar(x_forecast.flatten()[i], 
                         yhat[0].flatten()[i],
                         yerr=delta_f.flatten()[i],
                         c='k', capsize=3,
                         marker='o', ms=8, 
                         mec=color, mfc=color
                         )
            eb += [eb_add[0]]

        for x, y, error in zip(x_forecast, yhat[0], yhat[1]):
            display(Latex('$y(x={0:0.4f})={1:0.4f}\pm {2:0.4f}$'\
                          .format(x[0], y[0], error[0])))

        return eb 

    #Perform the regression in linear space unless we're plotting log data, as
    #opposed to plotting unadjusted data but on a log scale
    if xadjustment == 'log':
        reg_xscale = 'log'
    elif xadjustment is None:
        reg_xscale = 'linear'
    else:
        raise ValueError('Adjustment should be \'log\' or None')
    if yadjustment == 'log':
        reg_yscale = 'log'
    elif yadjustment is None:
        reg_yscale = 'linear'
    else:
        raise ValueError('Adjustment should be \'log\' or None')

    mlr_res = dm_den.mlr(fname, xcols=[xcol], 
                         ycol=ycol,
                         xscales=[reg_xscale], yscale=reg_yscale,
                         dropgals=dropgals, 
                         prediction_x=prediction_x, fore_sig=fore_sig, dX=dX,
                         verbose=verbose)
    coefs, intercept, r2, Xs, ys, ys_pred, r2a, resids = mlr_res[:8]
    if prediction_x is not None:
        prediction_y = mlr_res[-1] #[y, y uncertainty] 
        # ebc is an ErrorbarContainer. I think by telling adjust_texts to avoid
        # ebc[0], it will avoid the prediction point.
        eb = plt_forecast(ax, prediction_x, prediction_y)
        adjust_text_kwargs['add_objects'] = eb

    df = dm_den.load_data(fname)
    if dropgals:
        df = df.drop(dropgals)

    if xadjustment=='log':
        xlabel=loglabel(xlabel)
    if yadjustment=='log':
        ylabel=loglabel(ylabel)
    fill_ax_new(ax, df, xcol, ycol, 
                xlabel=xlabel,
                ylabel=ylabel, 
                xscale=xscale,
                yscale=yscale,
                xadjustment=xadjustment,
                yadjustment=yadjustment,
                showcorr=False,
                arrowprops=arrowprops, showlabels=showlabels, 
                minarrow=minarrow, adjust_text_kwargs=adjust_text_kwargs,
                labelsize=labelsize, **kwargs)
    ax.plot(Xs[0], ys_pred, label=legend_txt)

    if show_formula:
        formula_strings = {'vcirc_R0':'v_\mathrm{c}',
                           'v_cool_gas':'v_\\phi',
                           'disp_dm_solar':'\sigma_\mathrm{{DM}}', 
                           'den_solar':'\\rho_\mathrm{{DM}}'}
        
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

        if xadjustment=='log' and yadjustment=='log':
            #if plotting log data on both axes, show the formula of the form y=Ax^m
            
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
                            .format(ystring, coefs[0], xstring, np.abs(intercept), r2,
                                    operator)))
                display(Latex('$r^2={4:0.2f}$'
                            .format(ystring, coefs[0], xstring, np.abs(intercept), r2,
                                    operator)))
            else:
                ax.annotate('${0:s}={1:0.2f}{2:s}{5:s}{3:0.2f}$\n'
                            '$r^2={4:0.2f}$'\
                            .format(ystring, coefs[0], xstring, np.abs(intercept), r2,
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
        showGeV_y(ax, yadjustment)

    result = [coefs, intercept, resids]
    if prediction_x is not None:
        result += [prediction_y]
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
                showcorr=True, legend_txt=None, minarrow=0.02,
                adjust_text_kwargs={}, xtickspace=None, ytickspace=None):
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
    ax.plot(xs,ys,'o',color=color,alpha=alpha,label=legend_txt)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
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
            texts+=[ax.annotate(name, (float(x), float(y)), 
                                fontsize=labelsize)]
        if arrowprops:
            adjust_text(texts, arrowprops=arrowprops, ax=ax, 
                        **adjust_text_kwargs)
            ###################################################################
            # Remove short arrows
            ###################################################################
            for child in ax.get_children():
                if isinstance(child, mpl.text.Annotation):
                    arrowlen = np.linalg.norm(np.array(child.xy) \
                                              - np.array([child._x, 
                                                          child._y]))
                    if arrowlen < minarrow:
                        child.arrowprops=None
            ###################################################################
        else:
            adjust_text(texts, ax=ax)

    if xtickspace is not None:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=xtickspace))
    if ytickspace is not None:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=ytickspace))

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

def convert_log_errors(logyhat):
    log_yhat_min = logyhat[0]-logyhat[1]
    log_yhat_max = logyhat[0]+logyhat[1]
    yhat_min = 10.**log_yhat_min
    yhat_max = 10.**log_yhat_max
    dy = (yhat_max-yhat_min)/2.
    return dy

def make_err_bars_fr_resids(ax, reg):
    '''
    Make rror bars from residuals
    '''
    resids = reg[2]
    delta_neg = np.percentile(resids, (1.-0.682)/2.*100.)
    delta_pos = np.percentile(resids, (1.-(1.-0.682)/2.)*100.)
    delta = np.mean(np.abs((delta_neg, delta_pos)))
    
    ax.errorbar(np.log10(v0_sofue), 
                reg[-1][0], #The last element of reg will be the prediction.
                yerr=delta, 
                marker='o', ms=8, c='k', mec='r', mfc='r', capsize=3)
    
    return None

def draw_xshade(ax, ycol, v0, dv0):
    if ycol=='den_disc':
        bounds = (rho_min_sofue, rho_max_sofue)
        ax.axhspan(*bounds, 
                    alpha=0.2, color='gray', ls='none')
    return None

def draw_shades(ax, ycol, v0, dv0):

    draw_xshade(ax, ycol, v0, dv0)

    xlo = np.log10(v0-dv0)
    xhi = np.log10(v0+dv0)
    ax.axvspan(xlo, 
               xhi, 
               alpha=0.2, color='gray', ls='none')
    return None

def plt_vs_vc(ycol, tgt_fname, source_fname='dm_stats_20220715.h5',
              fore_sig=1.-0.682, verbose=False, minarrow=0.03,
              adjust_text_kwargs={}, show_formula='outside',
              figsize=(10,5), labelsize=14., v0=v0_eilers, dv0=dv0_eilers):

    df = dm_den.load_data(source_fname)
    textxy = (0.04, 0.96)
    fontsize = 14
    formula_y = -0.4
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if ycol=='den_disc':
        ylabel = den_label
    elif ycol=='disp_dm_disc_cyl':
        ylabel = disp_label

    reg_disc = ax_slr(ax,source_fname,
                     'v_dot_phihat_disc(T<=1e4)',
                     ycol,
                     xlabel=vc_label,
                     ylabel=ylabel,
                     xadjustment='log', yadjustment='log',
                     formula_y=formula_y, dropgals=['m12w','m12z'],
                     arrowprops={'arrowstyle':'-'}, 
                     show_formula=show_formula,
                     showlabels=True, 
                     prediction_x=[[v0]], fore_sig=fore_sig, 
                     dX=[[dv0]], showGeV=True, verbose=verbose,
                     minarrow=minarrow, adjust_text_kwargs=adjust_text_kwargs,
                     labelsize=labelsize, xtickspace=0.05, ytickspace=0.05)
    yhat_vc = reg_disc[-1]
    
    display(Latex('$r=8.3\pm{0:0.2f}\,\mathrm{{kpc}}$'
                  .format(df.attrs['dr']/2., df.attrs['dz']/2.)))
    display(Latex('$|z|\in[0,{1:0.2f}]\,\mathrm{{kpc}}$' \
                  .format(df.attrs['dr']/2., df.attrs['dz']/2.)))

    plt.draw()
    
    plt.savefig(paths.figures+tgt_fname,
                bbox_inches='tight',
                dpi=140)

    plt.show()
    
    return yhat_vc


def plt_vs_gmr_vc(ycol, tgt_fname=None, source_fname='dm_stats_20220715.h5',
                  fore_sig=1.-0.682, verbose=False, minarrow=0.03,
                  adjust_text_kwargs={}, show_formula='outside',
                  figsize=(10,5), labelsize=14., v0=v0_eilers, dv0=dv0_eilers):
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
    
    draw_shades(ax0, ycol, v0, dv0)
    draw_shades(ax1, ycol, v0, dv0)
    
    reg_gmr = ax_slr(ax0, 
                      source_fname,
                      'vcirc',
                      ycol,
                      xlabel=gmr_label, ylabel=ylabel,
                      xadjustment='log', yadjustment='log',
                      dropgals=['m12w','m12z'],
                      arrowprops={'arrowstyle':'-'}, 
                      show_formula=show_formula, prediction_x=[[v0]],
                      dX=[[dv0]], showGeV=False, 
                      showlabels=True, formula_y=formula_y, verbose=verbose,
                      minarrow=minarrow, adjust_text_kwargs=adjust_text_kwargs,
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
                     prediction_x=[[v0]], fore_sig=fore_sig, 
                     dX=[[dv0]], showGeV=True, verbose=verbose,
                     minarrow=minarrow, adjust_text_kwargs=adjust_text_kwargs,
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

def plt_disc_diffs(df_source='dm_stats_20220715.h5', 
                   diff_source='den_disp_dict_20220818.pkl',
                   only_linear=False, 
                   only_log=False, figsize=None, tgt_fname=None,
                   update_val=False):
    
    direc='/export/nfs0home/pstaudt/projects/project01/data/'
    with open(direc+diff_source, 'rb') as handle:
        den_disp_dict = pickle.load(handle)
    df = dm_den.load_data(df_source)
    galnames = df.drop(['m12w','m12z']).index
    #galnames = df.index
    
    def setup(log):
        if not log:
            denlabel = '$\\rho(\phi)/\,\overline{\\rho}$'
            displabel = '$\sigma(\phi)/\,\overline{\sigma}$'

            dens = np.array([den_disp_dict[galname]['dens/avg'] \
                             for galname in galnames]).flatten()
            disps = np.array([den_disp_dict[galname]['disps/avg'] \
                              for galname in galnames]).flatten()
        else:
            denlabel = '$\log\\rho(\phi)\,/\,\log\overline{\\rho}$'
            displabel = '$\log\sigma(\phi)\,/\,\log\overline{\sigma}$'

            dens = np.array([den_disp_dict[galname]['log(dens)/log(avg)'] \
                             for galname in galnames]).flatten()
            disps = np.array([den_disp_dict[galname]['log(disps)/log(avg)'] \
                              for galname in galnames]).flatten()
            percent_diff = staudt_utils.round_up(
                100.*np.max(np.abs(1.-np.array([dens,disps]))),
                3)
            if update_val:
                #update the value in data.dat for the paper
                dm_den.save_var_latex('maxdiff',
                                      '{0:0.2f}\%'.format(percent_diff))
            staudt_utils.print_eq('\min\Delta\log\\rho/\log\overline{\\rho}',
                     np.min(dens-1))
            staudt_utils.print_eq('\max\Delta\log\\rho/\log\overline{\\rho}',
                     np.max(dens-1))
            staudt_utils.print_eq('\min\Delta\log\\sigma/\log\overline{\\sigma}',
                     np.min(disps-1))
            staudt_utils.print_eq('\max\Delta\log\\sigma/\log\overline{\\sigma}',
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
    ylabel = '$N_\mathrm{\phi\,bin}$'
    
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

    if tgt_fname:
        plt.savefig(paths.figures+tgt_fname,
                    bbox_inches='tight',
                    dpi=140)

    plt.show()
    
    return None

def plt_gmr_vs_vc(df_source='dm_stats_20220715.h5', tgt_fname='gmr_vs_vc.png',
                  figsize=(8,4),
                  labelsize=11., minarrow=0.01, adjust_text_kwargs={}):
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
           minarrow=minarrow, legend_txt='best fit',
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

def plt_particle_counts(df_source='dm_stats_20220715.h5'):
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
        ax.grid()
        plt.show()

    return None

if __name__=='__main__':
    fname=sys.argv[1]
    df=dm_den.load_data(fname)
    #Take each dict of inputs from make_plot_feed() and make a plot with each:
    for f in make_plot_feed(df):
        plotter(**f)
