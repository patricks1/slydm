import scipy
import numpy as np

def gen_max(v,v0,a):
    f = 4.*np.pi*v**2.*np.exp(-(v/v0)**(2.*a))
    N = 4./3.*np.pi * v0**3. * scipy.special.gamma(1.+3./(2.*a))
    return f / N

def gen_gauss(x,mu,x0,a):
    y=-(((x-mu)/x0)**2.)**a
    f = np.exp(y)
    N = 2.*x0*scipy.special.gamma(1.+1./2./a)
    '''print(mu)
    print(x0)
    print('alpha={0:0.20f}'.format(a))'''
    return f/N

def trunc_max(v,v0,a,vesc):
    f = v**2.*np.exp(-(v/v0)**(2.*a))
    g1 = scipy.special.gamma(3./2./a)
    g2 = scipy.special.gammainc((v0/vesc)**(-2.*a),3./2./a)
    N = v0**3./2./a * (g1 - g2)
    p = f/N
    
    if isinstance(v,(list,np.ndarray)):
        isesc = v>=vesc
        p[isesc] = 0.
    else:
        if v>=vesc:
            return 0.
    return p

def fit_dim_gal(galname, dim, pdf_dict, ax, trunc=False, trunc_max=trunc_max):
    bins=pdf_dict[galname][dim]['bins']
    xs=(bins[1:]+bins[:-1])/2.
    xs_fit_plt = np.linspace(xs.min(),xs.max(),int(1e3))
    ys=pdf_dict[galname][dim]['ps']
    
    if dim=='mag':
        if trunc:
            res_gen, _ = scipy.optimize.curve_fit(trunc_max, xs, ys,
                                                  p0=[3.,1.,6.])
            v0_gen, a_gen, vesc_gen = res_gen
            ys_fit_gen = trunc_max(xs,*res_gen)
            ys_fit_gen_plt = trunc_max(xs_fit_plt,*res_gen)

            maxwell = lambda xs,v0: gen_max(xs,v0,1.)
            res, _ = scipy.optimize.curve_fit(maxwell, xs, ys)
            v0 = res[0]
            ys_fit = maxwell(xs,v0)
            ys_fit_plt = maxwell(xs_fit_plt,v0)
        else:
            res_gen, _ = scipy.optimize.curve_fit(gen_max, xs, ys,
                                                  p0=[10.,1.])
            v0_gen, a_gen, = res_gen
            ys_fit_gen = gen_max(xs,*res_gen)
            ys_fit_gen_plt = gen_max(xs_fit_plt,*res_gen)

            maxwell = lambda xs,v0: gen_max(xs,v0,1.)
            res, _ = scipy.optimize.curve_fit(maxwell, xs, ys)
            v0 = res[0]
            ys_fit = maxwell(xs,v0)
            ys_fit_plt = maxwell(xs_fit_plt,v0)
    elif dim in ['x','y','z']:
        p=Parameters()
        p.add('mu',value=0.,vary=False,min=-1.,max=1.,brute_step=0.01)
        p.add('x0',value=1.,vary=True,min=1e-6,max=6.,brute_step=0.01)
        p.add('a',value=1.,vary=True,min=-20.,max=20.)
        
        def errs_gen_gauss(p,xs,ys_data):
            mu=p['mu'].value
            x0=p['x0'].value
            a=p['a'].value
            ys_fit=gen_gauss(xs,mu,x0,a)
            errs=ys_data-ys_fit
            return errs
        
        res_min = minimize(errs_gen_gauss, p, args=(xs,ys),
                           method='lm')
        res_gen = [res_min.params[key].value for key in res_min.params]
        mu_gen, v0_gen, a_gen = res_gen
        ys_fit_gen = gen_gauss(xs,*res_gen)
        ys_fit_gen_plt = gen_gauss(xs_fit_plt,*res_gen)
    
        gauss = lambda xs,mu,v0: gen_gauss(xs,mu,v0,1.)
        res, _ = scipy.optimize.curve_fit(gauss, xs, ys)
        mu, v0 = res
        ys_fit = gauss(xs,*res)
        ys_fit_plt = gauss(xs_fit_plt,*res)
    else:
        raise ValueError('dim must be \'x\' \'y\' \'z\' \'mag\' or \'magtrunc\'')
    
    sse_gen = np.sum((ys-ys_fit_gen)**2.) #SSE for generalized distribution
    sser_gen=sse_gen/(xs.size-len(res_gen))    
    sse_1=np.sum((ys-ys_fit)**2.)
    sser_1=sse_1/(xs.size-len(res))
    
    ax.step(xs,ys,'k',where='mid',color='grey')
    ax.plot(xs_fit_plt,ys_fit_gen_plt,'-',color='b',lw=2.)
    ax.plot(xs_fit_plt,ys_fit_plt,'-',color='red')
    
    loc=[1.,.983]
    
    kwargs_txt = dict(fontsize=12., xycoords='axes fraction',
                  va='top', ha='right', 
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    if dim=='mag':
        if trunc:
            ax.annotate(galname,loc,
                        **kwargs_txt)
            txt_gen = '$\\xi_0={0:0.2f};$\n$\\alpha={1:0.2f};$\n'\
                      '$\\xi_\mathrm{{esc}}={3:0.1f}$\n'\
                      '$\mathrm{{SSE}}_\mathrm{{r}}={2:0.1e}$'.format(v0_gen,a_gen,sser_gen,vesc_gen)
            inv = ax.transAxes.inverted()
            loc=ax.transAxes.transform(loc)
            loc[1]-=17.
            loc=inv.transform(loc)
            ax.annotate(txt_gen, loc, color='b',
                        **kwargs_txt)
            txt_1 = '$\\xi_0={0:0.3f};$\n$\\alpha=1;$\n$\mathrm{{SSE}}_\mathrm{{r}}={1:0.1e}$'.format(v0,sser_1)
            loc=ax.transAxes.transform(loc)
            loc[1]-=75.
            loc=inv.transform(loc)
            ax.annotate(txt_1, loc,
                        color='r',
                        **kwargs_txt)          
        else:
            ax.annotate(galname,loc,
                        **kwargs_txt)
            txt_gen = '$\\xi_0={0:0.2f};$\n$\\alpha={1:0.2f};$\n'\
                      '$\mathrm{{SSE}}_\mathrm{{r}}={2:0.1e}$'.format(v0_gen,a_gen,sser_gen)
            inv = ax.transAxes.inverted()
            loc=ax.transAxes.transform(loc)
            loc[1]-=17.
            loc=inv.transform(loc)
            ax.annotate(txt_gen, loc, color='b',
                        **kwargs_txt)
            txt_1 = '$\\xi_0={0:0.3f};$\n$\\alpha=1;$\n$\mathrm{{SSE}}_\mathrm{{r}}={1:0.1e}$'.format(v0,sser_1)
            loc=ax.transAxes.transform(loc)
            loc[1]-=55.
            loc=inv.transform(loc)
            ax.annotate(txt_1, loc,
                        color='r',
                        **kwargs_txt)        
    elif dim in ['x','y','z']:
        ax.annotate(galname,loc,
                    **kwargs_txt)
        inv = ax.transAxes.inverted()
        loc=ax.transAxes.transform(loc)
        loc[1]-=20.
        loc=inv.transform(loc)
        txt_gen = '$\mu={3:0.3f};$\n'\
                  '$\\xi_0={0:0.2f};$\n$\\alpha={1:0.2f};$\n'\
                  '$\mathrm{{SSE}}_\mathrm{{r}}={2:0.2e}$\n'\
                  .format(v0_gen,a_gen,sser_gen,mu_gen)
        ax.annotate(txt_gen, loc, color='b',
                    **kwargs_txt)
        txt_1 = '$\mu={2:0.3f};$\n'\
                '$\\xi_0={0:0.2f};$\n$\\alpha=1;$\n$\mathrm{{SSE}}_\mathrm{{r}}={1:0.2e}$'\
                .format(v0,sser_1,mu)
        loc=ax.transAxes.transform(loc)
        loc[1]-=75.
        loc=inv.transform(loc)
        ax.annotate(txt_1, loc,
                    color='r',
                    **kwargs_txt)
    return sser_gen, a_gen
