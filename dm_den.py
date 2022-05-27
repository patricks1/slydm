import h5py
import os
import time
import pickle
import logging
import traceback
import numpy as np
import pandas as pd
from staudt_fire_utils import get_data, show
from sklearn.linear_model import LinearRegression
from progressbar import ProgressBar
from astropy import units as u
from astropy import constants as c
from matplotlib import pyplot as plt
from SHMpp.SHMpp_gvmin_reconcile import vE as vE_f
#import dm_den_viz

def build_direcs(gal, res, mass_class, typ='fire'):
    assert typ in ['fire','dmo']
    if typ=='fire':
        typ_char='B'
    elif typ=='dmo':
        typ_char='D'
    res=str(res)
    if mass_class>10:
        mass_class='{0:02d}'.format(mass_class)
        hdirec='/data17/grenache/aalazar/FIRE/GV'+typ_char+'/m'+mass_class+gal+\
               '_res'+res+\
               '/halo/rockstar_dm/hdf5/halo_600.hdf5'
        direc='/data17/grenache/aalazar/FIRE/GV'+typ_char+'/m'+mass_class+gal+'_res'+res+\
              '/output/hdf5/'
    elif mass_class==10:
        if typ=='dmo':
            raise ValueError('Cannot yet handle DMO for log M < 11')
        path = '/data25/rouge/mercadf1/FIRE/m10x_runs/' #Path to m10x runs
        run = 'h1160816' #input the run name
        haloName = 'm'+mass_class+gal #input the halo name within this run
        pt = 'PartType1' #You can change this to whatever particle you want
        hdirec=path+run+'/'+haloName+'/halo_pos.txt'
        direc=path+run+'/output/snapshot_'+run+'_Z12_bary_box_152.hdf5'
    else:
        raise ValueError('Cannot yet handle log M < 10')
    return hdirec, direc

def unpack_new(df, galname, dr=1.5, drsolar=None, typ='fire', 
               getparts=['PartType1']):
    suffix=df.loc[galname,'fsuffix']
    res=df.loc[galname,'res']
    host_key=df.loc[galname,'host_key']
    mass_class = df.loc[galname,'mass_class']
    halodirec, direc = build_direcs(suffix, res, mass_class, typ=typ)
    s=600
    s=str(s)
    sdir=direc+'snapdir_'+s+'/snapshot_'+s

    #Determine number of files
    N=len(os.listdir(direc+'snapdir_'+s+'/'))

    #Need to input N in the following line
    #even though we're just grabbing h, because if there are >1
    #files, the file name convention will reflect that.
    h=get_data(sdir,N,'Header','HubbleParam')

    # Getting host halo info
    p, rvir, v, mvir = get_halo_info(halodirec, suffix, typ, host_key, mass_class)
    
    for i,parttype in enumerate(getparts):
        print('Unpacking {0:s} data'.format(parttype))
        pbar=ProgressBar()
        for j in pbar(range(0,N)):
            if N==1:
                fname=sdir+'.hdf5'
            else:
                fname=sdir+'.'+str(j)+'.hdf5'
            with h5py.File(fname,'r') as f:
                ms_add=f[parttype]['Masses'][:] #in units of 1e10 M_sun / h
                ms_add/=h #now in units of 1e10 M_sun
                ms_add=ms_add.astype(np.longdouble)
                coords_add=f[parttype]['Coordinates'][:]
                coords_add/=h
                rs_add=np.linalg.norm(coords_add-p,axis=1)
                v_vecs_add=f[parttype]['Velocities'][:]
                parttypes_add=np.repeat(parttype,len(coords_add))
                #print(v_vecs_add)
                if i==0 and j==0:
                    coords=coords_add
                    rs=rs_add
                    ms=ms_add
                    v_vecs=v_vecs_add
                    parttypes=parttypes_add
                else:
                    coords=np.concatenate((coords,coords_add),axis=0)
                    rs=np.concatenate((rs,rs_add))
                    ms=np.concatenate((ms,ms_add))
                    #print(v_vecs)
                    #print(v_vecs_add)
                    v_vecs=np.concatenate((v_vecs,v_vecs_add),axis=0)
                    parttypes=np.concatenate((parttypes,parttypes_add))
                #print(np.sum( (rs<8.3+0.75)&(rs>8.3-0.75)))

    v_vecs-=v

    #rs=np.linalg.norm(coords-p,axis=1)
    v_mags=np.linalg.norm(v_vecs,axis=1)

    return ms, mvir, rs, rvir, v_mags, v_vecs, parttypes, coords

def unpack(df, galname, dr=1.5, drsolar=None, typ='fire', 
           getparts=['PartType1']):
    suffix=df.loc[galname,'fsuffix']
    res=df.loc[galname,'res']
    host_key=df.loc[galname,'host_key']
    mass_class = df.loc[galname,'mass_class']
    halodirec, direc = build_direcs(suffix, res, mass_class, typ=typ)
    s=600
    s=str(s)
    sdir=direc+'snapdir_'+s+'/snapshot_'+s

    #Determine number of files
    N=len(os.listdir(direc+'snapdir_'+s+'/'))

    #Need to input N in the following line
    #even though we're just grabbing h, because if there are >1
    #files, the file name convention will reflect that.
    h=get_data(sdir,N,'Header','HubbleParam')

    coords=[]
    ms=[]
    v_vecs=[]
    parttypes=[]

    for t in getparts:
        print('Unpacking {0:s} data'.format(t))
        coords_add=get_data(sdir,N,t,'Coordinates')
        coords_add/=h
        coords.extend(coords_add)
        ms_add=get_data(sdir,N,t,'Masses') #in units of 1e10 M_sun / h
        ms_add/=h #now in units of 1e10 M_sun
        ms_add=ms_add.astype(np.longdouble)
        ms.extend(ms_add)
        v_vecs_add=get_data(sdir,N,t,'Velocities')
        v_vecs.extend(v_vecs_add)
        parttypes_add=np.repeat(t,len(coords_add))
        parttypes.extend(parttypes_add)
    print('Converting to ndarrays')
    start=time.time()
    coords=np.array(coords)
    ms=np.array(ms)
    v_vecs=np.array(v_vecs)
    parttypes=np.array(parttypes)
    print('Elapsed time: {0:0.0f}s'.format(time.time()-start))

    # Getting host halo info
    p, r, v, mvir = get_halo_info(halodirec, suffix, typ, host_key, mass_class)
    
    v_vecs-=v
    rs=np.linalg.norm(coords-p,axis=1)
    v_mags=np.linalg.norm(v_vecs,axis=1)

    return ms, mvir, rs, r, v_mags, v_vecs, parttypes

def unpack_4pot(df, galname, typ='fire'):
    getparts=['PartType0','PartType1','PartType4']
    #getparts=['PartType1']
    suffix=df.loc[galname,'fsuffix']
    res=df.loc[galname,'res']
    host_key=df.loc[galname,'host_key']
    mass_class = df.loc[galname,'mass_class']
    halodirec, direc = build_direcs(suffix, res, mass_class, typ=typ)
    s=600
    s=str(s)
    sdir=direc+'snapdir_'+s+'/snapshot_'+s

    #Determine number of files
    N=len(os.listdir(direc+'snapdir_'+s+'/'))

    #Need to input N in the following line
    #even though we're just grabbing h, because if there are >1
    #files, the file name convention will reflect that.
    h=get_data(sdir,N,'Header','HubbleParam')

    # Getting host halo info
    p, rvir, v, mvir = get_halo_info(halodirec, suffix, typ, 
                                     host_key, mass_class)
    
    for i,parttype in enumerate(getparts):
        print('Unpacking {0:s} data'.format(parttype))
        pbar=ProgressBar()
        for j in pbar(range(0,N)):
            if N==1:
                fname=sdir+'.hdf5'
            else:
                fname=sdir+'.'+str(j)+'.hdf5'
            with h5py.File(fname,'r') as f:
                ms_add=f[parttype]['Masses'][:] #in units of 1e10 M_sun / h
                ms_add/=h #now in units of 1e10 M_sun
                ms_add=ms_add.astype(np.longdouble)
                coords_add=f[parttype]['Coordinates'][:]
                coords_add/=h
                coords_add-=p #puts coords in kpc centered on the halo
                parttypes_add=np.repeat(parttype,len(coords_add))
                if i==0 and j==0:
                    coords=coords_add
                    ms=ms_add
                    parttypes=parttypes_add
                else:
                    coords=np.concatenate((coords,coords_add),axis=0)
                    ms=np.concatenate((ms,ms_add))
                    parttypes=np.concatenate((parttypes,parttypes_add))
    f = {'pos':coords,'parttype':parttypes,'mass':ms}
    return f

def get_halo_info(halodirec, suffix, typ, host_key, mass_class):
    if mass_class>10: 
        with h5py.File(halodirec,'r') as f:
            isrj = suffix=='_elvis_RomeoJuliet'
            istl = suffix=='_elvis_ThelmaLouise'
            ishost2 = host_key=='host2.index'
            isdmo = typ=='dmo'
            if (isrj or (istl and isdmo)) and ishost2:
                #RomeoJuliet doesn't have a halo2.index key
                #ThelmaLouise doesn't have a halo2.index key in the dmo sims
                ms_hals=f['mass.vir'][:]
                is_sorted=np.argsort(ms_hals)
                i=is_sorted[-2]
            else:
                i=f[host_key][0]
            p=f['position'][i]
            r=f['radius'][i] #Is this virial radius?
            v=f['velocity'][i]
            mvir = f['mass.vir'][i]
    elif mass_class==10:
        #I think this is going to need correcting. Will prob give an error.
        hposList = np.loadtxt(path+run+'/'+haloName+'/halo_pos.txt')
        p = hposList[0]
        rvirList = np.loadtxt(path+run+'/'+haloName+'/halo_r_vir.txt')
        r = rvirList[0]
        mvir=np.nan #Don't have a saved value for mvir
    else:
        raise ValueError('Cannot yet handle log M < 10')
    return p, r, v, mvir

def unpack_mwithin(fname):
    #I think this might be an abandoned function?
    df=load_data(fname)
    for galname in df.index:
        suffix=df.loc[galname,'fsuffix']
        res=df.loc[galname,'res']
        mass_class=df.loc[galname,'mass_class']
        host_key=df.loc[galname,'host_key']
        halodirec, direc = build_direcs(suffix, res, mass_class, typ=typ)
        s=600
        s=str(s)
        sdir=direc+'snapdir_'+s+'/snapshot_'+s

        coord_halo, rvir, v_halo, mvir = get_halo_info(halodirec, suffix,
                                                       'fire', host_key,
                                                       mass_class)

        #Determine number of files
        N=len(os.listdir(direc+'snapdir_'+s+'/'))



        result=[]
        pbar=ProgressBar()
        parttypes=['PartType0','PartType1','PartType4']
        for i in pbar(range(0,N)):
            with h5py.File(sdir,'r') as f:
                ms=f[parttype]['Masses'] #in units of 1e10 M_sun / h
                rs=f[parttype]['Coordinates']

def get_disp_within(r,rs,v_mags):
    is_in=rs<r
    disp=np.std(v_mags[is_in])
    return disp

def get_den_disp(r,rs,dr,ms,v_mags,v_vecs):
    rmax=r+dr/2.
    rmin=r-dr/2.
    if rmin<0:
        raise ValueError('r={0:0.1f}. It doesn\'t make sense to set dr >'
                         '2x R0.'.format(r))

    is_in=(rs<rmax) & (rs>rmin)
    print('{0:0.0f} particles in the shell'.format(np.sum(is_in)))
    v=4./3.*np.pi*(rmax**3.-rmin**3.) #kpc^3
    mtot=ms[is_in].sum()
    den=mtot/v
    disp=np.std(v_mags[is_in])
    devs = v_vecs[is_in] - np.average(v_vecs[is_in], axis=0)
    #print(devs)
    sum_sq_devs=np.sum(devs**2.,axis=0)
    disp3d=np.sqrt(np.sum(sum_sq_devs)/np.sum(is_in))
    def disp3d_francisco_f():
        v_avg=v_vecs[is_in].mean(axis=0)
        diffs=v_vecs[is_in] - v_avg
        coord_var=np.var(diffs,axis=0)
        disp=np.sqrt(np.sum(coord_var))
        return disp
    disp3d_francisco=disp3d_francisco_f()
    #print(disp3d)
    #print(disp3d_francisco)
    assert np.allclose(disp3d,disp3d_francisco)
    return den*10.**10., disp3d

def get_disp_btwn(rmin,rmax,rs,v_mags):
    is_in=(rs<rmax)&(rs>rmin)
    disp=np.std(v_mags[is_in])
    return disp

def get_mbtw(rmin,rmax,rs,ms):
    is_btw = (rs<rmax) & (rs>rmin)
    mtot=ms[is_btw].sum()
    return mtot*10.**10.

def get_mwithin(r,rs,ms):
    is_in=rs<r
    mtot=ms[is_in].sum()
    return mtot*10.**10.

def plot_tests(df, galname, dr=1.5, drsolar=None):

    from matplotlib import pyplot as plt
    from dm_den_viz import log_rho_solar_label, disp_vir_label
    
    ms, mvir, rs, r, v_mags, v_vecs, parttypes = unpack(df, galname, dr, 
                                                        drsolar)
    rsolar=8.3

    drs=np.linspace(0.1,rsolar*2.,100)
    dens=np.log10([get_den_disp(rsolar,rs,dr,ms,
                                v_mags,v_vecs)[0] for dr in drs])
    fig=plt.figure(figsize=(7,5))
    ax=fig.add_subplot(111)
    ax.plot(drs,dens,'bo')
    ax.set_ylabel(log_rho_solar_label)
    ax.set_xlabel('$dr$')
    plt.show()

    drs=np.linspace(0.1,r*2.,250)
    disps=[get_den_disp(r,rs,dr,ms,v_mags,v_vecs)[1] for dr in drs]
    fig=plt.figure(figsize=(7,5))
    ax=fig.add_subplot(111)
    ax.plot(drs,disps,'bo')
    ax.set_ylabel(disp_vir_label)
    ax.set_xlabel('$dr$')
    plt.show()

    rs_plot=np.linspace(1.5/2.,rsolar*2.,50)
    dens=np.log10([get_den_disp(r,rs,dr,ms,v_mags,v_vecs)[0] \
                   for r in rs_plot])
    fig=plt.figure(figsize=(7,5))
    ax=fig.add_subplot(111)
    ax.plot(rs_plot,dens,'bo')
    ax.set_ylabel('log $\\rho(r\pm{0:0.1f}$ kpc)'.format(dr))
    ax.set_xlabel('$r$')
    plt.show()

    rs_plot=np.linspace(1.5/2.,r+300.,100)
    disps=[get_disp_within(r, rs, v_mags) for r in rs_plot]
    fig=plt.figure(figsize=(7,5))
    ax=fig.add_subplot(111)
    ax.plot(rs_plot,disps,'bo')
    ax.set_ylabel('$\sigma(<r)$')
    ax.set_xlabel('$r$')
    plt.show()

    disps=[get_den_disp(r, rs, dr, ms, v_mags, v_vecs)[1] for r in rs_plot]
    fig=plt.figure(figsize=(7,5))
    ax=fig.add_subplot(111)
    ax.plot(rs_plot,disps,'bo')
    ax.set_xlabel('$r$')
    ax.set_ylabel('$\sigma(r\pm{0:0.1f}$ kpc)'.format(dr/2.))
    plt.show()

    rmaxs=np.linspace(r-100.+1.,r+100.,150)
    disps=[get_disp_btwn(r-100.,rmax,rs,v_mags) for rmax in rmaxs]
    fig=plt.figure(figsize=(7,5))
    ax=fig.add_subplot(111)
    ax.plot(rmaxs,disps,'bo')
    ax.set_ylabel('$\sigma(100\,\mathrm{kpc}<r<r_\mathrm{max})$')
    ax.set_xlabel('$r_\mathrm{max}$')
    plt.show()

    return None

def get_vcirc(r_axis,rs,ms):
    is_within = rs<r_axis
    mwithin = np.sum(ms[is_within])
    #Masses are in units of 1e10 M_sun
    vcirc = np.sqrt(c.G*mwithin*u.M_sun*1.e10 / (r_axis*u.kpc)).to(u.km/u.s).value
    return vcirc

def get_v_esc(ms, coords, rvec):
    '''
    Calculate escape velocity at rvec, given masses and the 
    coordinate locations of
    those masses
    '''
    d_pos = rvec - coords
    rs = np.linalg.norm(d_pos, axis=1)
    m_by_r = ms / rs
    pot = -c.G * m_by_r.sum() * 10.**10. * u.M_sun/u.kpc
    v_esc = np.sqrt(2.*np.abs(pot))
    return v_esc.to(u.km/u.s)

def analyze(df, galname, dr=1.5, drsolar=None, typ='fire',
            vcircparts=['PartType0','PartType1','PartType4']):
    if drsolar is None:
        drsolar=dr

    dat  = unpack_new(df, galname, dr, drsolar, typ=typ, getparts=vcircparts)
    ms_all, mvir, rs_all, r, v_mags_all, v_vecs_all, parttypes_all, _ = dat

    isdm = parttypes_all=='PartType1'
    print('{0:0d} dm particles'.format(np.sum(isdm)))
    ms = ms_all[isdm]
    rs = rs_all[isdm]
    v_mags = v_mags_all[isdm]
    v_vecs = v_vecs_all[isdm]

    for col in ['mvir_fromhcat','mvir_calc','m10tovir',
                'mwithin10','mvir_check']:
        if col not in df:
            df[col]=np.empty(len(df),dtype=np.longdouble)

    rsolar=8.3

    den_solar, disp_dm_solar = get_den_disp(rsolar, rs, dr=drsolar, 
                                            ms=ms, v_mags=v_mags, 
                                            v_vecs=v_vecs)
    df.loc[galname,['den_solar','disp_dm_solar']] = den_solar, disp_dm_solar 
    df.loc[galname,'f'] = 10.**7./den_solar
    df.loc[galname,['den_vir','disp_dm_vir']] = get_den_disp(r, rs, 
                                                             dr, ms, 
                                                             v_mags, v_vecs)
    isgas=parttypes_all=='PartType0'
    df.loc[galname,'disp_gas_solar'] = get_den_disp(rsolar, rs_all[isgas], dr,
                                                    ms_all[isgas], 
                                                    v_mags_all[isgas],
                                                    v_vecs_all[isgas])[1]
    df.loc[galname,'mvir_fromhcat'] = mvir
    df.loc[galname,'mvir_calc'] = get_mwithin(r,rs,ms)
    df.loc[galname,'m10tovir'] = get_mbtw(10.,r,rs,ms)
    df.loc[galname,'mwithin10'] = get_mwithin(10.,rs,ms)
    df.loc[galname,'mvir_check'] = df.loc[galname,
                                          'mwithin10'] + df.loc[galname,
                                                                'm10tovir']
    df.loc[galname,'vcirc_R0'] = get_vcirc(rsolar,rs_all,ms_all)
    return None

def init_df(mass_class=12):
    if not isinstance(mass_class,(int,str)):
        raise ValueError('mass_class should be an integer or \'all\'.')
    gals_m9 = ['']
    ress_m9 = [250]
    gal_names_m9 = ['m9']
    host_keys_m9 = ['host.index']
    mass_classs_m9 = [9]

    '''
    gals_m10 = ['b','c','d','e','f','g','h','i','j','k','l','m','q','v','w',
                'xa','xb','xc','xd','xe','xf','xg','xh','xi',
                'y','z']
    ress_m10 = [500]*12 + [250]*2 + [2100] + [4000]*9 + [2100]*2
    '''
    gals_m10 = ['q','v','w','y','z']
    #ress_m10 = [500]*12 + [250]*2 + [2100] + [4000]*9 + [2100]*2
    ress_m10 = [250]*2 + [2100]*3
    gal_names_m10=['m10' + g for g in gals_m10]
    host_keys_m10 = ['host.index']*len(gals_m10)
    mass_classs_m10 = [10]*len(gals_m10)
    assert len(gals_m10) == len(ress_m10) == len(host_keys_m10) \
           == len(gal_names_m10) == len(mass_classs_m10)

    #gals_m11 = ['a','b','c','d','e','f','g','h','i','q','v']
    #ress_m11 = [2100]*3 + [7100]*2 + [17000]*2 + [7100]*4
    #f and g don't have halo files, so I'm skipping them for now.
    gals_m11 = ['a','b','c','d','e','h','i','q','v']
    ress_m11 = [2100]*3 + [7100]*2 + [7100]*4
    gal_names_m11 = ['m11' + g for g in gals_m11]
    host_keys_m11 = ['host.index']*len(gals_m11)
    mass_classs_m11 = [11]*len(gals_m11)
    assert len(gals_m11) == len(ress_m11) == len(host_keys_m11) \
           == len(gal_names_m11) == len(mass_classs_m11)

    gals_m12 = ['b','c','f','i','m','r','w','z',
                '_elvis_RomeoJuliet',
                '_elvis_RomeoJuliet',
                '_elvis_RomulusRemus',
                '_elvis_RomulusRemus',
                '_elvis_ThelmaLouise',
                '_elvis_ThelmaLouise']
    ress_m12 = [7100,7100,7100,7100,7100,7100,7100,4200,
                3500,3500,4000,4000,4000,4000]
    gal_names_m12=['m12' + g for g in gals_m12[:8]] + ['Romeo','Juliet',
                                                       'Romulus','Remus',
                                                       'Thelma','Louise']
    host_keys_m12 = ['host.index']*8+['host.index','host2.index']*3
    mass_classs_m12 = [12]*len(gals_m12)
    assert len(gals_m12) == len(ress_m12) == len(host_keys_m12) \
           == len(gal_names_m12) == len(mass_classs_m12)

    if mass_class in [9,10,11,12]:
        lcls=locals()
        exec('gals = gals_m'+str(mass_class),globals(),lcls)
        exec('ress = ress_m'+str(mass_class),globals(),lcls)
        exec('gal_names = gal_names_m'+str(mass_class),globals(),lcls)
        exec('host_keys = host_keys_m'+str(mass_class),globals(),lcls)
        exec('mass_classs = mass_classs_m'+str(mass_class),globals(),lcls)
        gals=lcls['gals']
        ress=lcls['ress']
        gal_names=lcls['gal_names']
        host_keys=lcls['host_keys']
        mass_classs=lcls['mass_classs']
    elif mass_class=='all':
        '''
        gals = gals_m9 + gals_m10 + gals_m11 + gals_m12
        ress = ress_m9 + ress_m10 + ress_m11 + ress_m12
        gal_names = gal_names_m9 + gal_names_m10 + gal_names_m11 + gal_names_m12
        host_keys = host_keys_m9 + host_keys_m10 + host_keys_m11 + host_keys_m12
        '''
        #Removing m9 and m10 for now
        gals = gals_m11 + gals_m12
        ress = ress_m11 + ress_m12
        gal_names = gal_names_m11 + gal_names_m12
        host_keys = host_keys_m11 + host_keys_m12
        mass_classs = mass_classs_m11 + mass_classs_m12
    else:
        raise ValueError('mass_class must be 9,10,11,12 or \'all\'')
        
    df=pd.DataFrame(index=gal_names,
                    data=np.transpose([gals,ress,host_keys]),
                    columns=['fsuffix','res','host_key'])
    #Need to add mass_classs *after* creating df because we need it to have an 
    #int dtype.
    df['mass_class']=mass_classs 

    return df

def gen_data(fname=None, mass_class=12, dr=1.5, drsolar=None, typ='fire',
             vcircparts=['PartType0','PartType1','PartType4']):
    def insert_analysis(df):
        dat=[]
        for g in df.index:
            print('Retrieving '+g)
            analyze(df, g, dr=dr, drsolar=drsolar, typ=typ, 
                    vcircparts=vcircparts)
        return df 
    if not drsolar:
        drsolar=dr
    df = init_df(mass_class)
    df = df[df.index=='m12b']
    df = insert_analysis(df)
    df.dr=dr
    df.drsolar=drsolar
    if fname:
        direc='/export/nfs0home/pstaudt/projects/project01/data/'
        fname=direc+fname
        #df.to_hdf(fname,index=True,key='df')
        with pd.HDFStore(fname) as store:
            store.put('data',df)
            store.get_storer('data').attrs.metadata={'dr':dr,'drsolar':drsolar}
    return df
                    
def load_data(fname):
    direc='/export/nfs0home/pstaudt/projects/project01/data/'
    fname=direc+fname
    try:
        with pd.HDFStore(fname) as store:
            df=store['data']
            df.attrs=store.get_storer('data').attrs.metadata
    except KeyError:
        df=pd.read_hdf(fname)
    return df

def calc_f_vals(df):
    fs = 10.**7./df['den_solar']
    fs.name='f_vals'
    return fs

def comp_disp_vc(galname='m12i',dr=1.5,fname=None):
    df=init_df(mass_class=12)
    dat_fire = unpack_new(df, galname, typ='fire', 
                          getparts=['PartType0','PartType1','PartType4'])
    ms_fire, mvir_fire, rs_fire, r_fire, v_mags_fire, v_vecs_fire,\
        parttypes_fire, _ = dat_fire
    dat_dmo = unpack_new(df, galname, typ='dmo', 
                         getparts=['PartType1'])
    ms_dmo, mvir_dmo, rs_dmo, r_dmo, v_mags_dmo, v_vecs_dmo, \
        parttypes_dmo, _ = dat_dmo

    def run(r_axis, rs, dr, ms, v_mags, v_vecs, parttypes):
        ispart = parttypes=='PartType1' 
        _, disp = get_den_disp(r_axis, rs[ispart], dr, ms[ispart], 
                               v_mags[ispart], 
                               v_vecs[ispart])
        vcirc = get_vcirc(r_axis,rs,ms)
        return disp, vcirc
    N=20
    rs_axis=np.logspace(np.log10(dr/2.),np.log10(150.),N)
    disps_fire = np.zeros(N)
    vcircs_fire = np.zeros(N)
    disps_dmo = np.zeros(N)
    vcircs_dmo = np.zeros(N)
    print('Running calculations')
    for i, r_axis in enumerate(rs_axis):
        disps_fire[i], vcircs_fire[i] = run(r_axis, rs_fire, 
                                            dr, ms_fire, 
                                            v_mags_fire, 
                                            v_vecs_fire,
                                            parttypes_fire)
        disps_dmo[i], vcircs_dmo[i] = run(r_axis, rs_dmo, dr, ms_dmo, 
                                          v_mags_dmo, 
                                          v_vecs_dmo,
                                          parttypes_dmo)
    return ms_fire, ms_dmo, rs_fire, rs_dmo, vcircs_fire, vcircs_dmo
    if fname:
        direc='/export/nfs0home/pstaudt/projects/project01/data/'
        fname=direc+fname
        with h5py.File(fname,'w') as f:
            for name, data in zip(['rs','disps_fire','vcircs_fire',
                                   'disps_dmo','vcircs_dmo'],
                                  [rs_axis,disps_fire,vcircs_fire,disps_dmo,
                                   vcircs_dmo]):
                f.create_dataset(name,data=data)

    return rs_axis, disps_fire, vcircs_fire, disps_dmo, vcircs_dmo 

def v_pdf(df, galname, bins=50, r=8.3, dr=1.5, incl_ve=False):
    ms, mvir, rs, rvir, v_mags, v_vecs, parttypes, _ = unpack_new(df, galname)
    if incl_ve:
        if 'disp_dm_solar' in df and not np.isnan(df.loc[galname,'disp_dm_solar']):
            sigma3d = df.loc[galname,'disp_dm_solar'] 
        else:
            den_solar, sigma3d = get_den_disp(r, rs, dr=dr, 
                                              ms=ms, v_mags=v_mags, 
                                              v_vecs=v_vecs)
            df.loc[galname,['den_solar','disp_dm_solar']] = den_solar, sigma3d 
    v0 = sigma3d*np.sqrt(2./3.)
    print(v0)
    day = 60.8 #Gives avg val of ve
    ve = np.array(vE_f(day, v0))
    print(ve)
    v_vecs = v_vecs - ve
    v_mags = np.linalg.norm(v_vecs,axis=1)

    inshell=(rs<r+dr/2.)&(rs>r-dr/2.)
    print(v_mags[:100])
    print(v_mags[inshell].min(),v_mags[inshell].max())
    ps, bins, _ = plt.hist(v_mags[inshell], bins=bins, density=True)
    mu = np.average(v_mags[inshell])
    plt.close()
    return ps, bins, mu

def make_v_pdfs(bins=50, r=8.3, dr=1.5, fname=None, incl_ve=False):
    df=init_df() 
    pdfs={}

    for galname in df.index:
        print('Generating {0:s}'.format(galname))

        pdfs[galname]={}
        res=v_pdf(df, galname, incl_ve=incl_ve)
        pdfs[galname]['ps'],pdfs[galname]['bins'],pdfs[galname]['v_avg'] = res
    if fname:
        direc='/export/nfs0home/pstaudt/projects/project01/data/'
        fname=direc+fname
        with open(fname,'wb') as f:
            pickle.dump(pdfs, f, pickle.HIGHEST_PROTOCOL)
    return pdfs

def x_3d_pdf(df, galname, bins=100, r=8.3, dr=1.5, test_unpack=None, 
             typ='fire'):
    if test_unpack:
        ms, mvir, rs, rvir, v_mags, v_vecs, parttypes = test_unpack
    else:
        ms, mvir, rs, rvir, v_mags, v_vecs, parttypes, _ = unpack_new(df, 
                                                                      galname,
                                                                      typ=typ)
    v_vecs=v_vecs.reshape(3,-1)
    inshell=(rs<r+dr/2.)&(rs>r-dr/2.)
    d={}
    d['x']={}
    d['y']={}
    d['z']={}
    d['mag']={}

    def hist_dim(vs, d):
        assert vs.ndim==1
        sigma=np.std(vs)
        xs=vs/sigma
        d['ps'], d['bins'], _ = plt.hist(xs[inshell], bins=bins, 
                                         density=True)
        plt.close()
        return None

    for vs, ddim in zip([v_vecs[0], v_vecs[1], v_vecs[2], v_mags],
                        [d['x'], d['y'], d['z'], d['mag']]):
        hist_dim(vs, ddim)
    return d 

def x_pdf(df, galname, bins=50, r=8.3, dr=1.5):
    ms, mvir, rs, rvir, v_mags, v_vecs, parttypes, _ = unpack_new(df, galname)
    sigma=df.loc[galname,'disp_solar']
    xs=v_mags/sigma
    inshell=(rs<r+dr/2.)&(rs>r-dr/2.)
    ps, bins, _ = plt.hist(xs[inshell], bins=bins, density=True)
    plt.close()
    return ps, bins

def make_x_3d_pdfs(bins=100, r=8.3, dr=1.5, fname=None, 
                   dffile='dm_den_20210804_2.h5', typ='fire'):
    #df=load_data(dffile)
    df=init_df()
    pdfs={}
    for galname in df.index:
        print('Generating {0:s}'.format(galname))
        pdfs[galname] = x_3d_pdf(df,galname,typ=typ)
    if fname:
        direc='/export/nfs0home/pstaudt/projects/project01/data/'
        fname=direc+fname
        with open(fname,'wb') as f:
            pickle.dump(pdfs, f, pickle.HIGHEST_PROTOCOL)
    return pdfs

def make_x_pdfs(bins=50, r=8.3, dr=1.5, fname=None, 
                dffile='dm_den_20210804_2.h5'):
    df=load_data(dffile)
    pdfs={}
    for galname in df.index:
        print('Generating {0:s}'.format(galname))
        pdfs[galname]={}
        pdfs[galname]['ps'],pdfs[galname]['bins'] = x_pdf(df,galname)
    if fname:
        direc='/export/nfs0home/pstaudt/projects/project01/data/'
        fname=direc+fname
        with open(fname,'wb') as f:
            pickle.dump(pdfs, f, pickle.HIGHEST_PROTOCOL)
    return pdfs

def mlr(fsource, xcols, ycol, xscales=None, yscale='log'):
    '''
    Multi-linear regression
    '''
    assert isinstance(xcols,(list,tuple,np.ndarray))
    df=load_data(fsource)
    ys=df[ycol]
    if yscale=='log':
        ys=np.log10(ys)
    if xscales:
        assert len(xcols)==len(xscales)
        Xs=np.empty((len(df),len(xcols)))
        assert Xs.shape==df[xcols].shape
        Xs=Xs.transpose()
        for i, vals in enumerate(zip(xcols, xscales)):
            xcol, scale = vals
            if scale == 'log':
                Xs[i]=np.log10(df[xcol])
            elif scale == 'linear':
                Xs[i]=df[xcol]
            else:
                raise ValueError('Scale should be \'log\' or \'linear\'')
        Xs=Xs.transpose()
    else:
        Xs=df[xcols]

    model=LinearRegression()
    model.fit(Xs, ys)
    r2=model.score(Xs, ys)
    coefs=model.coef_
    intercept=model.intercept_
    ys_pred = model.predict(Xs)

    Xs=Xs.transpose()

    return coefs, intercept, r2, Xs, ys, ys_pred

def get_v_escs(fname=None):
    '''
    Get escape velocities for all M12's, without any pre-existing analysis
    '''

    def fill_v_escs_dic(gal):
        dic = unpack_4pot(df, gal)
        
        thetas = np.pi*np.array([1./4., 1./2., 3./4., 1., 5./4., 3./2., 7./4., 2.])
        #thetas = np.pi*np.array([1./2., 1., 3./2., 2.])
        v_escs[gal] = {}
        v_escs[gal]['v_escs'] = []
        
        print('Calculating v_e:')
        pbar = ProgressBar()
        for theta in pbar(thetas):
            rvec = r * np.array([np.cos(theta), np.sin(theta), 0.])
            v_escs[gal]['v_escs'] += [get_v_esc(dic['mass'], dic['pos'], rvec)]
        print('')
        v_escs[gal]['std_ve'] = np.std(v_escs[gal]['v_escs'])
        v_escs[gal]['ve_avg'] = np.mean(v_escs[gal]['v_escs'])

        return None

    v_escs = {}
    df = init_df()
    r = 8.3 #kpc

    try:
        if fname:
            direc='/export/nfs0home/pstaudt/projects/project01/data/'
            fname=direc+fname
            f = open(fname,'wb')
        for gal in df.index:
            print('Analyzing {0:s}'.format(gal))
            fill_v_escs_dic(gal)
        
        if fname:
                pickle.dump(v_escs, f, pickle.HIGHEST_PROTOCOL)
                f.close()
    except KeyboardInterrupt as e:
        if fname:
            f.close()
            print('File closed: {0:s}'.format(str(f.closed)))
        raise e
    except Exception as e:
        if fname:
            f.close()
            print('File closed: {0:s}'.format(str(f.closed)))
        raise e 
    return v_escs

if __name__=='__main__':
    gen_data(fname='dm_den_20210623_2.h5')
