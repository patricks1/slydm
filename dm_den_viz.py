from scipy.stats.stats import pearsonr
from adjustText import adjust_text
import numpy as np
import pandas as pd
import dm_den
import sys
import staudt_utils
import staudt_fire_utils as utils

from astropy import units as u
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

def plt_slr(fname, xcol, ycol,
            xlabel,ylabel,
            xadjustment=None, yadjustment=None,
            xscale='linear', yscale='linear', 
            figsize=(7,6), dpi=100,
            showlabels=True,
            labelsize=15, arrowprops=None, formula_y=-0.2,
            dropgals=None):
    'Plot a simple linear regression'

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    ax_slr(ax, fname, xcol, ycol,
            xlabel,ylabel,
            xadjustment, yadjustment,
            xscale, yscale, 
            showlabels,
            labelsize, arrowprops, formula_y,
            dropgals)

    plt.show()

def ax_slr(ax, fname, xcol, ycol,
            xlabel,ylabel,
            xadjustment=None, yadjustment=None,
            xscale='linear', yscale='linear', 
            showlabels=True,
            labelsize=15, arrowprops=None, formula_y=-0.2,
            dropgals=None, showGeV=True):
    'Plot a simple linear regression on ax'

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
                         dropgals=dropgals)
    coefs, intercept, r2, Xs, ys, ys_pred = mlr_res

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
                xadjustment=xadjustment,
                yadjustment=yadjustment,
                showcorr=False,
                arrowprops=arrowprops)
    ax.plot(Xs[0], ys_pred)

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
        amplitude_str = staudt_utils.mprint(10.**intercept,
                                            d=1,
                                            show=False).replace('$','')
        xstring, ystring = get_strs()
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

    return coefs, intercept

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
                showcorr=True, legend_txt=None):
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

if __name__=='__main__':
    fname=sys.argv[1]
    df=dm_den.load_data(fname)
    #Take each dict of inputs from make_plot_feed() and make a plot with each:
    for f in make_plot_feed(df):
        plotter(**f)
