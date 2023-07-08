import h5py
import os
import csv
import time
import pickle
import logging
import traceback
import scipy
import math
import tabulate
import paths
import copy
import numpy as np
import pandas as pd
from staudt_fire_utils import get_data, show
from sklearn.linear_model import LinearRegression
from progressbar import ProgressBar
from astropy import units as u
from astropy import constants as c
from matplotlib import pyplot as plt
from SHMpp.SHMpp_gvmin_reconcile import vE as vE_f

def build_direcs(suffix, res, mass_class, typ='fire', source='original',
                 min_radius=None, max_radius=None):
    assert typ in ['fire','dmo']
    if typ=='fire':
        if source == 'cropped':
            # Using this folder while I make sure I'm not breaking anything by
            # discontinuing the use of float128's
            typ_char='B_202304'
            #print('type_char: {0:s}'.format(typ_char))
        else:
            typ_char='B'
    elif typ=='dmo':
        typ_char='D'

    res=str(res)
    mass_class='{0:02d}'.format(mass_class)
    snapnum = str(600)

    if source=='original':
        if min_radius is not None or max_radius is not None:
            raise ValueError('radius limits are not applicable when '
                             'source=\'original\'')
        topdirec = '/data17/grenache/aalazar/'
        cropstr = ''
    elif source=='cropped':
        if min_radius is None or max_radius is None:
            raise ValueError('min and max radii must be specified if '
                             'source=\'cropped\'')
        topdirec = '/data17/grenache/staudt/'
        cropstr = '{0:0.1f}_to_{1:0.1f}_kpc/'.format(min_radius, max_radius)
    else:
        raise ValueError('source should be \'original\' or \'cropped\'')
    if int(mass_class)>10:
        hdirec='/data17/grenache/aalazar/FIRE/GV'+typ_char+\
               '/m'+mass_class+suffix+\
               '_res'+res+\
               '/halo/rockstar_dm/hdf5/halo_600.hdf5'
        direc = topdirec+'FIRE/GV'+typ_char+'/m'+mass_class+suffix+'_res'+res+\
                '/output/'+cropstr+'hdf5/'
    elif int(mass_class)==10:
        raise ValueError('Cannot yet handle log M < 11')
        #The following code will not work, but I've leaving it for future
        #development
        if typ=='dmo':
            raise ValueError('Cannot yet handle DMO for log M < 11')
        path = '/data25/rouge/mercadf1/FIRE/m10x_runs/' #Path to m10x runs
        run = 'h1160816' #input the run name
        haloName = 'm'+mass_class+suffix #input the halo name within this run
        pt = 'PartType1' #You can change this to whatever particle you want
        hdirec=path+run+'/'+haloName+'/halo_pos.txt'
        direc=path+run+'/output/snapshot_'+run+'_Z12_bary_box_152.hdf5'
    else:
        raise ValueError('Cannot yet handle log M < 10')

    snapdir = direc+'snapdir_'+snapnum+'/'
    try:
        num_files=len(os.listdir(snapdir))
    except:
        num_files=None
    #path to the snapshot directory PLUS the first part of the filename:
    almost_full_path = snapdir+'snapshot_'+snapnum

    return hdirec, snapdir, almost_full_path, num_files

def build_direcs_old(suffix, res, mass_class, typ='fire', source='original'):
    assert typ in ['fire','dmo']
    if typ=='fire':
        typ_char='B'
    elif typ=='dmo':
        typ_char='D'
    res=str(res)
    mass_class='{0:02d}'.format(mass_class)
    if source=='original':
        topdirec = '/data17/grenache/aalazar/'
    elif source=='cropped':
        topdirec = '/data17/grenache/staudt/'
    else:
        raise ValueError('source should be \'original\' or \'cropped\'')
    if int(mass_class)>10:
        hdirec='/data17/grenache/aalazar/FIRE/GV'+typ_char+'/m'+mass_class+suffix+\
               '_res'+res+\
               '/halo/rockstar_dm/hdf5/halo_600.hdf5'
        direc = topdirec+'/FIRE/GV'+typ_char+'/m'+mass_class+suffix+'_res'+res+\
                '/output/hdf5/'
    elif int(mass_class)==10:
        raise ValueError('Cannot yet handle log M < 11')
        #The following code will not work, but I've leaving it for future
        #development
        if typ=='dmo':
            raise ValueError('Cannot yet handle DMO for log M < 11')
        path = '/data25/rouge/mercadf1/FIRE/m10x_runs/' #Path to m10x runs
        run = 'h1160816' #input the run name
        haloName = 'm'+mass_class+suffix #input the halo name within this run
        pt = 'PartType1' #You can change this to whatever particle you want
        hdirec=path+run+'/'+haloName+'/halo_pos.txt'
        direc=path+run+'/output/snapshot_'+run+'_Z12_bary_box_152.hdf5'
    else:
        raise ValueError('Cannot yet handle log M < 10')
    return hdirec, direc

def unpack_new(df, galname, dr=1.5, drsolar=None, typ='fire', 
               getparts=['PartType1']):
    '''
    Pull the galaxy's data from the original hdf5 files.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing necessary information about each galaxy.
        df_init() can generate a DataFrame with the correct information to
        make the pull.
    galname: str
        The galaxy name string corresponding to an index in df.
    dr: float
        The thickness of the spherical shell where the program will
        perform analysis
    drsolar: float, default: dr
        The thickness of the spherical shell for analysis in the solar 
        region. If drsolar is None, setting drsolar=dr is handled by
        analysis methods.
    typ: str: {'fire', 'dmo'}
        Specifies whether to pull FIRE or DMO data
    getparts: list of str: {'PartType0' : 'PartType4'}
        Specifies the particle types to extract

    Returns
    -------
    ms: np.ndarray, shape=(number of particles,)
        Physical masses in units of 1e10 M_sun
    mvir: float
        Virial mass from the halo file in units of 1e10 M_sun
    rs: np.ndarray, shape=(number of particles,)
        Each particle's radial distances from the center of the galaxy, in 
        units of kpc
    rvir: float
        Virial mass from the halo file in units of 1e10 M_sun
    v_mags: np.ndarray, shape=(number of particles,)
        Magnitudes of velocity vectors centered on the galaxy in units of km/s
    v_vecs: np.ndarray, shape=(number of particles, 3)
        Velocity vectors centered on the galaxy in units of km/s
    parttypes: np.array of str
        Particle types
    coords: np.array, shape=(number of particles, 3)
        Physical coordinate vectors, not rotated, centered on the galaxy
    p: np.ndarray, shape=(3,)
        Physical coordinate of the galaxy center, relative to the simulation,
        from the halo file, in units
        of kpc
    ''' 

    suffix=df.loc[galname,'fsuffix']
    res=df.loc[galname,'res']
    host_key=df.loc[galname,'host_key']
    mass_class = df.loc[galname,'mass_class']
    halodirec, direc = build_direcs_old(suffix, res, mass_class, typ=typ)
    s=600
    s=str(s)
    #path to the snapshot directory PLUS the first part of the filename:
    almost_full_path=direc+'snapdir_'+s+'/snapshot_'+s

    #Determine number of files
    N=len(os.listdir(direc+'snapdir_'+s+'/'))

    #Need to input N in the following line
    #even though we're just grabbing h, because if there are >1
    #files, the file name convention will reflect that.
    h=get_data(almost_full_path,N,'Header','HubbleParam')

    # Getting host halo info 
    p, rvir, v, mvir = get_halo_info(halodirec, suffix, typ, host_key, 
                                     mass_class)
    
    for i,parttype in enumerate(getparts):
        print('Unpacking {0:s} data'.format(parttype))
        pbar=ProgressBar()
        for j in pbar(range(0,N)):
            if N==1:
                fname=almost_full_path+'.hdf5' 
            else:
                fname=almost_full_path+'.'+str(j)+'.hdf5'
            with h5py.File(fname,'r') as f:
                ms_add=f[parttype]['Masses'][:] #in units of 1e10 M_sun / h
                ms_add/=h #now in units of 1e10 M_sun
                # Commenting the following because of the problems longdoubles cause on
                # MacOS
                #ms_add=ms_add.astype(np.longdouble)
                coords_add=f[parttype]['Coordinates'][:]
                coords_add/=h
                rs_add=np.linalg.norm(coords_add-p,axis=1)
                v_vecs_add=f[parttype]['Velocities'][:]
                parttypes_add=np.repeat(parttype,len(coords_add))
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
                    v_vecs=np.concatenate((v_vecs,v_vecs_add),axis=0)
                    parttypes=np.concatenate((parttypes,parttypes_add))

    v_vecs-=v

    v_mags=np.linalg.norm(v_vecs,axis=1)

    return ms, mvir, rs, rvir, v_mags, v_vecs, parttypes, coords, p

def unpack_gas(df, galname, typ='fire'):
    getparts=['PartType0']
    suffix=df.loc[galname,'fsuffix']
    res=df.loc[galname,'res']
    host_key=df.loc[galname,'host_key']
    mass_class = df.loc[galname,'mass_class']
    halodirec, direc = build_direcs_old(suffix, res, mass_class, typ=typ)
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
    p, rvir, v, mvir = get_halo_info(halodirec, suffix, typ, host_key, 
                                     mass_class)
    
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
                # Commenting the following because of the problems longdoubles cause on
                # MacOS
                #ms_add=ms_add.astype(np.longdouble)
                coords_add=f[parttype]['Coordinates'][:]
                coords_add/=h
                rs_add=np.linalg.norm(coords_add-p,axis=1)
                v_vecs_add=f[parttype]['Velocities'][:]
                parttypes_add=np.repeat(parttype,len(coords_add))
                energy_add = f[parttype]['InternalEnergy'][:]
                e_abundance_add = f[parttype]['ElectronAbundance'][:]
                he_frac_add = f[parttype]['Metallicity'][:,1]
                if i==0 and j==0:
                    coords=coords_add
                    rs=rs_add
                    ms=ms_add
                    v_vecs=v_vecs_add
                    parttypes=parttypes_add
                    energy = energy_add
                    e_abundance = e_abundance_add
                    he_frac = he_frac_add
                else:
                    coords=np.concatenate((coords,coords_add),axis=0)
                    rs=np.concatenate((rs,rs_add))
                    ms=np.concatenate((ms,ms_add))
                    v_vecs=np.concatenate((v_vecs,v_vecs_add),axis=0)
                    parttypes=np.concatenate((parttypes,parttypes_add))
                    energy = np.concatenate((energy,energy_add))
                    e_abundance = np.concatenate((e_abundance, 
                                                  e_abundance_add))
                    he_frac = np.concatenate((he_frac, he_frac_add))
    v_vecs-=v
    v_mags=np.linalg.norm(v_vecs,axis=1)

    return ms, mvir, rs, rvir, v_mags, v_vecs, parttypes, coords, energy,\
           e_abundance, he_frac

def calc_temps(he_fracs, e_abundances, energies):
    y_hes = he_fracs / (4.*(1.-he_fracs))
    mus = (1.+4.*y_hes) / (1+y_hes+e_abundances)
    mean_molecular_weights = mus * c.m_p
    gamma = 5./3. #adiabatic constant
    Ts = mean_molecular_weights.si * (gamma-1.) * energies*u.km**2./u.s**2. \
         / c.k_B
    return Ts.to(u.K)

def unpack(df, galname, dr=1.5, drsolar=None, typ='fire', 
           getparts=['PartType1']):
    suffix=df.loc[galname,'fsuffix']
    res=df.loc[galname,'res']
    host_key=df.loc[galname,'host_key']
    mass_class = df.loc[galname,'mass_class']
    halodirec, direc = build_direcs_old(suffix, res, mass_class, typ=typ)
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
        # Commenting the following because of the problems longdoubles cause on
        # MacOS
        #ms_add=ms_add.astype(np.longdouble)
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
    '''
    Returns
    -------
    data: dict 
        data['pos']: np.ndarray, shape = (number of particles, 3)
                Cartesian coordinates centered on the halo, in physical kpc
        data['parttype']: np.ndarray of str
            Particle type.  'PartType0' is gas. 'PartType1' is dark matter.
            'PartType2' is dummy collisionless. 'PartType3' is grains/PIC
            particles. 'PartType4' is stars. 'PartType5' is black holes / 
            sinks.
        data['mass']: np.ndarray 
            Mass of each particle in physical units of 1e10 Msun
    '''
    getparts=['PartType0','PartType1','PartType4']
    suffix=df.loc[galname,'fsuffix']
    res=df.loc[galname,'res']
    host_key=df.loc[galname,'host_key']
    mass_class = df.loc[galname,'mass_class']
    halodirec, direc = build_direcs_old(suffix, res, mass_class, typ=typ)
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
                # Commenting the following line that turns masses into
                # longdoubles, because of the problems it causes on MacOS.
                #ms_add=ms_add.astype(np.longdouble)
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
    data = {'pos':coords,'parttype':parttypes,'mass':ms}
    return data

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
        halodirec, direc = build_direcs_old(suffix, res, mass_class, typ=typ)
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
    #Note, this is not the right way to get dispersion. Maybe delete this.
    is_in=rs<r
    disp=np.std(v_mags[is_in])
    return disp

def get_den_disp(r, rs, dr, ms, v_mags, v_vecs, zs=None, dz=None, phis=None,
                 phi_bin=None, verbose=True):
    '''
    This works correctly (3D disp)
    The function doesn't use v_mags, but I'm keeping it in for now so nothing
    breaks.
    '''
    rmax=r+dr/2.
    rmin=r-dr/2.
    if rmin<0:
        raise ValueError('r={0:0.1f}. It doesn\'t make sense to set dr >'
                         '2x R0.'.format(r))

    inshell=(rs<rmax) & (rs>rmin)
    if dz is not None:
        indisc = np.abs(zs) < dz/2.
        def calc_vol(r,z):
            '''
            Volume of a sphere with its caps chopped off @ z and -z
            '''
            v = 2.*np.pi * (r**2.*z - z**3./3.)
            return v
        #Volume of the thick shell wth caps chopped off @ dz/2 and -dz/2:
        v = calc_vol(rmax,dz/2.) - calc_vol(rmin,dz/2.)
        if phi_bin is not None:
            # If we're only looking at a certain phi range, limit the volume to
            # that range
            fraction = (phi_bin[1] - phi_bin[0]) / (2.*np.pi)
            v *= fraction
            in_phi_bin = (phis >= phi_bin[0]) & (phis < phi_bin[1])
        else:
            in_phi_bin = np.repeat(True, len(inshell))
    else:
        indisc = np.repeat(True, len(inshell))
        in_phi_bin = np.repeat(True, len(inshell))
        v=4./3.*np.pi*(rmax**3.-rmin**3.) #kpc^3
    if verbose:
        print('{0:0.0f} particles in the ring'.format(np.sum(inshell \
                                                              & indisc)))
        print('{0:0.0f} particles in the slice'.format(np.sum(inshell \
                                                              & indisc \
                                                              & in_phi_bin)))
    mtot=ms[inshell & indisc & in_phi_bin].sum()
    den=mtot/v
    devs = v_vecs[inshell & indisc & in_phi_bin] \
           - np.average(v_vecs[inshell & indisc & in_phi_bin], axis=0)
    sum_sq_devs=np.sum(devs**2.,axis=0)
    disp3d=np.sqrt(np.sum(sum_sq_devs)/np.sum(inshell & indisc & in_phi_bin))
    def disp3d_francisco_f():
        v_avg=v_vecs[inshell & indisc & in_phi_bin].mean(axis=0)
        diffs=v_vecs[inshell & indisc & in_phi_bin] - v_avg
        coord_var=np.var(diffs,axis=0)
        disp=np.sqrt(np.sum(coord_var))
        return disp
    disp3d_francisco=disp3d_francisco_f()
    try:
        assert np.allclose(disp3d,disp3d_francisco)
    except AssertionError as e:
        print('my dispersion method: {0:0.2f}'.format(disp3d))
        print('francisco\'s dispersion method: {0:0.2f}'
              .format(disp3d_francisco))
        print('The dispersion methods don\'t match.')
        raise e
    return den*10.**10., disp3d

def atan(y,x):
    atan_val = math.atan2(y,x)
    if atan_val<0.:
        atan_val += 2.*np.pi
    return atan_val

def den_disp_phi_bins(source_fname, tgt_fname=None, N_slices=15, 
                      verbose=False):
    '''
    Generate a dictionary of density and dispersion in phi slices, split up by
    galaxy
    '''
    import cropper

    df = load_data(source_fname)
    pbar = ProgressBar()
    d = {} #initialize dictionary
    phi_slices = np.linspace(0., 2.*np.pi, N_slices+1)
    for k, galname in enumerate(pbar(df.index)):
        gal = cropper.load_data(galname, getparts=['PartType1'], verbose=False)
        phis = np.array([atan(y,x) \
                         for x,y in gal['PartType1']['coord_rot'][:,:2]])

        v_vec_cyls = np.transpose([gal['PartType1'][key] \
                                   for key in ['v_dot_rhat',
                                               'v_dot_phihat',
                                               'v_dot_zhat']],
                                  (1,0))

        d[galname] = {}
        d[galname]['dens'] = []
        d[galname]['disps'] = []
        d[galname]['phi_bins'] = phi_slices.copy()
        dens = d[galname]['dens']
        disps = d[galname]['disps']

        zs = gal['PartType1']['coord_rot'][:,2]
        for i, phi_bin in enumerate([phi_slices[j:j+2] 
                                     for j in range(N_slices)]):
            # phi_bin is a 2-element list. The 0 element is the beginning edge
            # of the bin. The 1 element is the ending edge of the bin.
            #using a copy just to make sure phi_bin doesn't get modified
            bin_ = phi_bin.copy() 
            if i == N_slices-1:
                # Add a small value to the ending edge of the bin, which is 
                # 2pi. We do this in case any particles are *exactly* at 2pi;
                # otherwise we would miss them.
                bin_[1] = bin_[1]+1.e-5 
            den_add, disp_add = get_den_disp(8.3,
                                             gal['PartType1']['r'],
                                             df.attrs['drsolar'],
                                             gal['PartType1']['mass_phys'],
                                             v_mags = None,
                                             v_vecs = v_vec_cyls,
                                             zs = zs,
                                             dz = df.attrs['dz'],
                                             phis = phis,
                                             phi_bin = bin_,
                                             verbose=verbose)
            dens+=[den_add]
            disps+=[disp_add]                        
        d[galname]['dens/avg'] = dens/df.loc[galname,'den_disc']
        d[galname]['log(dens)/log(avg)'] = np.log10(dens) \
                                           / np.log10(df.loc[galname,
                                                             'den_disc'])
        #max_den_diff = np.max(max_den_diff,
        #                      np.abs(1.-d[galname]['log(dens)/log(avg)']))
        d[galname]['disps/avg'] = disps/df.loc[galname,'disp_dm_disc_cyl']
        d[galname]['log(disps)/log(avg)'] = np.log10(disps) \
                                       / np.log10(df.loc[galname,
                                                         'disp_dm_disc_cyl'])
    if tgt_fname:
        direc = paths.data
        fname=direc+tgt_fname
        with open(fname,'wb') as f:
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    return d

def get_disp_btwn(rmin,rmax,rs,v_mags):
    # This is isometric (i.e. wrong). Consider deleting it. 
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

def calc_vesc(ms, coords, rvec):
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
            vcircparts=['PartType0','PartType1','PartType4'],
            source='original', dz=0.5):
    
    from cropper import load_data as load_cropped
    from cropper import flatten_particle_data

    if drsolar is None:
        drsolar=dr

    rsolar=8.3

    if source=='original':
        dat  = unpack_new(df, galname, dr, drsolar, typ=typ, 
                          getparts=vcircparts)
        ms_all, mvir, rs_all, rvir, v_mags_all, v_vecs_all, \
            parttypes_all = dat[:7]

        isdm = parttypes_all=='PartType1'
        print('{0:0d} dm particles'.format(np.sum(isdm)))
        ms_dm = ms_all[isdm]
        rs_dm = rs_all[isdm]
        v_mags_dm = v_mags_all[isdm]
        v_vecs_dm = v_vecs_all[isdm]

        isgas=parttypes_all=='PartType0'
        rs_gas = rs_all[isgas]
        ms_gas = ms_all[isgas]
        v_vecs_gas = v_vecs_all[isgas]
        v_mags_gas = v_mags_all[isgas]

        isstellar = parttypes_all=='PartType4'
        rs_stellar = rs_all[isstellar]
        ms_stellar = ms_all[isstellar]

    elif source=='cropped':
        d = load_cropped(galname)

        '''
        # Getting host halo info ##############################################
        suffix=df.loc[galname,'fsuffix']
        res=df.loc[galname,'res']
        host_key=df.loc[galname,'host_key']
        mass_class = df.loc[galname,'mass_class']

        halodirec = build_direcs(suffix, res, mass_class, typ)[0]
        rvir = get_halo_info(halodirec, suffix, typ, host_key, 
                             mass_class)[1]
        #######################################################################
        '''
        
        ms_dm = d['PartType1']['mass_phys']
        rs_dm = d['PartType1']['r']
        v_vecs_dm = d['PartType1']['v_vec_centered']
        #velocity vectors in cylindrical coordinates
        v_vecs_dm_cyl = np.array([d['PartType1']['v_dot_rhat'],
                               d['PartType1']['v_dot_phihat'],
                               d['PartType1']['v_dot_zhat']]).transpose(1,0)
        v_mags_dm = np.linalg.norm(v_vecs_dm, axis=1)
        zs = d['PartType1']['coord_rot'][:,2]

        rs_gas = d['PartType0']['r']
        ms_gas = d['PartType0']['mass_phys']
        v_vecs_gas = d['PartType0']['v_vec_centered']
        v_mags_gas = np.linalg.norm(v_vecs_gas, axis=1)
        v_vecs_cyl_gas = np.array([d['PartType0']['v_dot_rhat'],
                                   d['PartType0']['v_dot_phihat'],
                                   d['PartType0']['v_dot_zhat']]).transpose(1,
                                                                            0)
        zs_gas = d['PartType0']['coord_rot'][:,2]

        rs_all = flatten_particle_data(d, 'r')
        ms_all = flatten_particle_data(d, 'mass_phys')

        # limited to the disc
        den_disc, disp_dm_disc = get_den_disp(rsolar, rs_dm, dr,
                                              ms=ms_dm, v_mags=None,
                                              v_vecs=v_vecs_dm_cyl,
                                              zs=zs, dz=dz, verbose=False)

        df.loc[galname,'den_disc'] = den_disc
        df.loc[galname,'f_disc'] = 10.**7./den_disc
        df.loc[galname,'disp_dm_disc_cyl'] = disp_dm_disc

        # cylindrical coords, in a shell not limited to the disc
        _, disp_dm_cyl = get_den_disp(rsolar, rs_dm, dr=drsolar, ms=ms_dm, 
                                            v_mags=None, v_vecs=v_vecs_dm_cyl,
                                            zs=None, dz=None, verbose=False)
        df.loc[galname,'disp_dm_shell_cyl'] = disp_dm_cyl

        #######################################################################
        # Get avg v dot phihat of cool gas
        #######################################################################
        cooler1e4 = d['PartType0']['T'] < 1.e4
        cooler1e3 = d['PartType0']['T'] < 1.e3
        inshell = np.abs(d['PartType0']['r'] - rsolar) <= dr/2.
        indisc = np.abs(d['PartType0']['coord_rot'][:,2]) <= dz/2.
        df.loc[galname,
               'v_dot_phihat_disc(T<=1e4)'] = np.mean(d['PartType0']\
                                                       ['v_dot_phihat']\
                                                       [cooler1e4
                                                        & inshell
                                                        & indisc])
        df.loc[galname, 'vc100'] = df.loc[galname, 
                                          'v_dot_phihat_disc(T<=1e4)'] / 100.
        print(df.loc[galname, 'vc100'])
        df.loc[galname,
               'v_dot_phihat_shell(T<=1e4)'] = np.mean(d['PartType0']\
                                                        ['v_dot_phihat']\
                                                        [cooler1e4
                                                         & inshell])
        df.loc[galname,
               'v_dot_phihat_disc(T<=1e3)'] = np.mean(d['PartType0']\
                                                       ['v_dot_phihat']\
                                                       [cooler1e3
                                                        & inshell
                                                        & indisc])
        df.loc[galname,
               'v_dot_phihat_shell(T<=1e3)'] = np.mean(d['PartType0']\
                                                        ['v_dot_phihat']\
                                                        [cooler1e3
                                                         & inshell])
        #######################################################################

        df.loc[galname,
               'disp_gas_disc(T<1e4)'] = get_den_disp(
                                              rsolar, 
                                              rs_gas[cooler1e4],
                                              dr,
                                              ms_gas[cooler1e4],
                                              v_mags=None,
                                              v_vecs=v_vecs_cyl_gas[cooler1e4],
                                              zs=zs_gas[cooler1e4], 
                                              dz=dz, verbose=False)[1]
        df.loc[galname,
                'std(v_dot_phihat_disc(T<=1e4))'] = np.std(v_vecs_cyl_gas[:,1]\
                                                                [inshell
                                                                 & indisc
                                                                 & cooler1e4])
        df.loc[galname,
                'std(v_dot_phihat_shell(T<=1e4))'] = np.std(
                                                           v_vecs_cyl_gas[:,1]\
                                                                [inshell
                                                                 & cooler1e4])

    else:
        raise ValueError('source should be \'original\' or \'cropped\'')

    # "cart" means "Cartesian coords"
    if 'PartType1' in vcircparts:
        den_shell, disp_dm_cart = get_den_disp(rsolar, rs_dm, dr=drsolar, 
                                               ms=ms_dm, v_mags=v_mags_dm, 
                                               v_vecs=v_vecs_dm, verbose=False)
        df.loc[galname,['den_shell',
                        'disp_dm_shell_cart']] = den_shell, \
                                                 disp_dm_cart

        df.loc[galname,'f_shell'] = 10.**7./den_shell
    df.loc[galname,'vcirc'] = get_vcirc(rsolar,rs_all,ms_all)
     
    # Commenting the following because of the problems longdoubles cause on
    # MacOS
    #for col in ['mwithin10']: # Set the necessary columns to longdouble
    #    if col not in df:
    #        df[col]=np.empty(len(df),dtype=np.longdouble)
    df.loc[galname,'mwithin10'] = get_mwithin(10.,rs_dm,ms_dm)
    
    if source=='original':
        # Commenting the following because of the problems longdoubles cause on
        # MacOS
        #for col in ['mvir_fromhcat','mvir_calc','m10tovir',
        #            'mvir_check']:
        #    if col not in df:
        #        df[col]=np.empty(len(df),dtype=np.longdouble)

        if 'PartType0' in vcircparts:
            df.loc[galname,'disp_gas'] = get_den_disp(rsolar, rs_gas, dr,
                                                      ms_gas, 
                                                      v_mags_gas,
                                                      v_vecs_gas)[1]
        if 'PartType1' in vcircparts:
            df.loc[galname,['den_vir','disp_dm_vir']] = get_den_disp(rvir, 
                                                                     rs_dm, 
                                                                     dr, ms_dm, 
                                                                     v_mags_dm, 
                                                                     v_vecs_dm)
        df.loc[galname,'mvir_fromhcat'] = mvir
        df.loc[galname,'mvir_calc'] = get_mwithin(rvir,rs_dm,ms_dm)
        df.loc[galname,'mvir_stellar'] = get_mwithin(rvir, rs_stellar, 
                                                     ms_stellar)
        df.loc[galname,'m10tovir'] = get_mbtw(10.,rvir,rs_dm,ms_dm)
        df.loc[galname,'mvir_check'] = df.loc[galname,
                                              'mwithin10'] + df.loc[galname,
                                                                    'm10tovir']

    return None

def init_df(mass_class=12):
    if not isinstance(mass_class,(int,str)):
        raise ValueError('mass_class should be an integer or \'all\'.')
    suffixes_m9 = ['']
    # It's necessary to have different suffix lists for cropped and not cropped
    # because for cropped, I separate the RJ, RR, RL pair folders.
    suffixes_m9cropped = suffixes_m9.copy() 
    ress_m9 = [250]
    gal_names_m9 = ['m9']
    host_keys_m9 = ['host.index']
    mass_classs_m9 = [9]

    '''
    suffixes_m10 = ['b','c','d','e','f','g','h','i','j','k','l','m','q','v','w',
                'xa','xb','xc','xd','xe','xf','xg','xh','xi',
                'y','z']
    ress_m10 = [500]*12 + [250]*2 + [2100] + [4000]*9 + [2100]*2
    '''
    suffixes_m10 = ['q','v','w','y','z']
    suffixes_m10cropped = suffixes_m10.copy()
    #ress_m10 = [500]*12 + [250]*2 + [2100] + [4000]*9 + [2100]*2
    ress_m10 = [250]*2 + [2100]*3
    gal_names_m10=['m10' + g for g in suffixes_m10]
    host_keys_m10 = ['host.index']*len(suffixes_m10)
    mass_classs_m10 = [10]*len(suffixes_m10)
    assert len(suffixes_m10) == len(ress_m10) == len(host_keys_m10) \
           == len(gal_names_m10) == len(mass_classs_m10)

    #suffixes_m11 = ['a','b','c','d','e','f','g','h','i','q','v']
    #ress_m11 = [2100]*3 + [7100]*2 + [17000]*2 + [7100]*4
    #f and g don't have halo files, so I'm skipping them for now.
    suffixes_m11 = ['a','b','c','d','e','h','i','q','v']
    suffixes_m11cropped = suffixes_m11.copy()
    ress_m11 = [2100]*3 + [7100]*2 + [7100]*4
    gal_names_m11 = ['m11' + g for g in suffixes_m11]
    host_keys_m11 = ['host.index']*len(suffixes_m11)
    mass_classs_m11 = [11]*len(suffixes_m11)
    assert len(suffixes_m11) == len(ress_m11) == len(host_keys_m11) \
           == len(gal_names_m11) == len(mass_classs_m11)

    suffixes_m12 = ['b','c','f','i','m','r','w','z',
                '_elvis_RomeoJuliet',
                '_elvis_RomeoJuliet',
                '_elvis_RomulusRemus',
                '_elvis_RomulusRemus',
                '_elvis_ThelmaLouise',
                '_elvis_ThelmaLouise']
    suffixes_m12cropped = ['b','c','f','i','m','r','w','z',
                '_elvis_Romeo',
                '_elvis_Juliet',
                '_elvis_Romulus',
                '_elvis_Remus',
                '_elvis_Thelma',
                '_elvis_Louise']
    ress_m12 = [7100,7100,7100,7100,7100,7100,7100,4200,
                3500,3500,4000,4000,4000,4000]
    gal_names_m12=['m12' + g for g in suffixes_m12[:8]] + ['Romeo','Juliet',
                                                       'Romulus','Remus',
                                                       'Thelma','Louise']
    host_keys_m12 = ['host.index']*8+['host.index','host2.index']*3
    mass_classs_m12 = [12]*len(suffixes_m12)
    assert len(suffixes_m12) == len(ress_m12) == len(host_keys_m12) \
           == len(gal_names_m12) == len(mass_classs_m12)

    if mass_class in [9,10,11,12]:
        lcls=locals()
        exec('suffixes = suffixes_m'+str(mass_class),globals(),lcls)
        exec('suffixes_cropped = suffixes_m'+str(mass_class)+'cropped',
             globals(),lcls)
        exec('ress = ress_m'+str(mass_class),globals(),lcls)
        exec('gal_names = gal_names_m'+str(mass_class),globals(),lcls)
        exec('host_keys = host_keys_m'+str(mass_class),globals(),lcls)
        exec('mass_classs = mass_classs_m'+str(mass_class),globals(),lcls)
        suffixes=lcls['suffixes']
        suffixes_cropped = lcls['suffixes_cropped']
        ress=lcls['ress']
        gal_names=lcls['gal_names']
        host_keys=lcls['host_keys']
        mass_classs=lcls['mass_classs']
    elif mass_class=='all':
        '''
        suffixes = suffixes_m9 + suffixes_m10 + suffixes_m11 + suffixes_m12
        suffixes_cropped = suffixes_m9cropped + suffixes_m10cropped \
                           + suffixes_m11cropped + suffixes_m12cropped
        ress = ress_m9 + ress_m10 + ress_m11 + ress_m12
        gal_names = gal_names_m9 + gal_names_m10 + gal_names_m11 + gal_names_m12
        host_keys = host_keys_m9 + host_keys_m10 + host_keys_m11 + host_keys_m12
        '''
        #Removing m9 and m10 for now
        suffixes = suffixes_m11 + suffixes_m12
        suffixes_cropped = suffixes_m11cropped + suffixes_m12cropped
        ress = ress_m11 + ress_m12
        gal_names = gal_names_m11 + gal_names_m12
        host_keys = host_keys_m11 + host_keys_m12
        mass_classs = mass_classs_m11 + mass_classs_m12
    else:
        raise ValueError('mass_class must be 9,10,11,12 or \'all\'')
        
    df=pd.DataFrame(index=gal_names,
                    data=np.transpose([suffixes, suffixes_cropped, 
                                       ress, host_keys]),
                    columns=['fsuffix','fsuffix_cropped','res','host_key'])
    #Need to add mass_classs *after* creating df because we need it to have an 
    #int dtype.
    df['mass_class']=mass_classs 

    return df

def gen_data(fname=None, mass_class=12, dr=1.5, drsolar=None, typ='fire',
             vcircparts=['PartType0','PartType1','PartType4'], 
             source='original', dz=0.5, freq_save=False):
    def insert_analysis(df):
        #dat=[]
        for g in df.index:
            print('Retrieving '+g)
            analyze(df, g, dr=dr, drsolar=drsolar, typ=typ, 
                    vcircparts=vcircparts, source=source, dz=dz)
            if freq_save and fname:
                save_data(df, fname)
        return df 
    if not drsolar:
        drsolar=dr
    df = init_df(mass_class)
    #df = df[df.index=='m12b']
    df = insert_analysis(df)
    df.attrs = {}
    df.attrs['dr'] = dr
    df.attrs['drsolar'] = drsolar
    df.attrs['dz'] = dz
    if fname:
        save_data(df, fname)
    return df
                    
def convert_den(dens):
    '''
    Convert a pd.Series of densities in Msun / kpc^3 to GeV / cm^3
    '''
    dens = copy.deepcopy(dens)
    dens = dens.values
    dens = dens * u.Msun / u.kpc**3. * c.c**2.
    dens = dens.to(u.GeV / u.cm**3.)
    return dens

def test_gen_data(test_fname=None):
    df_old = load_data('dm_stats_dz1.0_20230616.h5')
    if test_fname is not None:
        df_new = load_data(test_fname)
    else:
        df_new = gen_data(source='cropped')

    changed_cols = []
    for col in df_old.columns:
        if col in df_new.columns:
            old = df_old[col]
            new = df_new[col]
            unchanged = True # initializing the variable
            if 'den' in col:
                gev_old = convert_den(old).value
                gev_new = convert_den(new).value
                unchanged = np.allclose(gev_old, gev_new)
                old = np.log10(old)
                new = np.log10(new)
            if old.dtype == 'object':
                unchaged = unchanged and np.array_equal(old, new)
            else:
                unchanged = unchaged and np.allclose(old, new)
            if not unchanged:
                changed_cols += [col]
                print('\n{0:s} failed'.format(col))
                print( np.array( [old, new] ).T )
    assert len(changed_cols) == 0
    print('gen_data test passed.')
    return None

def compare_dfs(fname1, fname2):
    df1 = load_data(fname1)
    df2 = load_data(fname2)

    for col in df1.columns:
        if col in df2.columns:
            one = df1[col]
            two = df2[col]
            if 'den' in col:
                gev_one = convert_den(one).value
                gev_two = convert_den(two).value
                unchanged = np.allclose(gev_one, gev_two)
                if not unchanged:
                    print('\n{0:s} has changed.'.format(col))
                    print( np.array( [gev_one, gev_two] ).T )
                one = np.log10(one)
                two = np.log10(two)
            if one.dtype == 'object':
                unchanged = np.array_equal(one, two)
            else:
                unchanged = np.allclose(one, two)
            if not unchanged:
                print('\n{0:s} has changed.'.format(col))
                print( np.array( [one, two] ).T )
    return None

def save_data(df, fname):
    abspath = os.path.abspath(__file__) #path to this script
    direc = os.path.dirname(abspath) #path to this script's directory
    direc += '/data/'
    fname=direc+fname
    with pd.HDFStore(fname) as store:
        store.put('data',df)
        store.get_storer('data').attrs.metadata = df.attrs
    return None

def load_data(fname):
    abspath = os.path.abspath(__file__) #path to this script
    direc = os.path.dirname(abspath) #path to this script's directory
    direc += '/data/'
    fname=direc+fname

    # Ensure the file exists
    if not os.path.isfile(fname):
        raise FileNotFoundError(fname)

    try:
        with pd.HDFStore(fname) as store:
            df=store['data']
            df.attrs=store.get_storer('data').attrs.metadata
    except (KeyError, ValueError):
        print(fname)
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
            parttypes_fire, = dat_fire[:7]
    dat_dmo = unpack_new(df, galname, typ='dmo', 
                         getparts=['PartType1'])
    ms_dmo, mvir_dmo, rs_dmo, r_dmo, v_mags_dmo, v_vecs_dmo, \
            parttypes_dmo = dat_dmo[:7]

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

def v_pdf(df, galname, bins=50, r=8.3, dr=1.5, incl_ve=False, dz=0.5, 
          density=True):
    import cropper
    d = cropper.load_data(galname, getparts=['PartType1'], verbose=False)
    dm = d['PartType1']
    ms = dm['mass_phys']
    rs = dm['r']
    # Cartesian coordinates rotated so zhat is aligned with the disc's angular
    # mementum
    coords = dm['coord_rot'] 
    zs = coords[:,2]
    # Velocity vectors in cylindrical coordinates
    v_vecs = np.array([dm['v_dot_rhat'], dm['v_dot_phihat'],
                           dm['v_dot_zhat']]).T
    v_mags = np.linalg.norm(v_vecs, axis=1)
    #ms, mvir, rs, rvir, v_mags, v_vecs, parttypes = unpack_new(df, galname)[:7]
    if incl_ve:
        if 'disp_dm_solar' in df and not np.isnan(df.loc[galname,
                                                         'disp_dm_solar']):
            sigma3d = df.loc[galname,'disp_dm_solar'] 
        else:
            den_solar, sigma3d = get_den_disp(r, rs, dr=dr, 
                                              ms=ms, v_mags=v_mags, 
                                              v_vecs=v_vecs)
            df.loc[galname,['den_solar','disp_dm_solar']] = den_solar, sigma3d 
        # I think we might want to use vc instead of v0 (peak speed estimate).
        # Maybe consider changing the code.
        v0 = sigma3d*np.sqrt(2./3.)
        day = 60.8 #Gives avg val of v_earth
        v_earth = np.array(vE_f(day, v0))
        v_vecs = v_vecs - v_earth
    v_mags = np.linalg.norm(v_vecs,axis=1)

    inshell=(rs<r+dr/2.)&(rs>r-dr/2.)
    if dz is not None:
        inshell = inshell & ( np.abs(zs) <= dz / 2. )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ps, bins, _ = ax.hist(v_mags[inshell], bins=bins, density=density)
    mu = np.average(v_mags[inshell])
    plt.close()
    return ps, bins, mu

def make_v_pdfs(bins=50, r=8.3, dr=1.5, fname=None, incl_ve=False, dz=0.5,
                density=True):
    df=init_df() 
    pdfs={}

    for galname in df.index:
        print('Generating {0:s}'.format(galname))

        pdfs[galname]={}
        res = v_pdf(df, galname, incl_ve=incl_ve, dz=dz, density=density)
        bins = res[1]
        vs = (bins[1:] + bins[:-1]) / 2.
        pdfs[galname]['vs'] = vs
        if density:
            pdfs[galname]['ps'] ,\
                    pdfs[galname]['bins'] ,\
                    pdfs[galname]['v_avg'] = res
        else:
            pdfs[galname]['counts'] ,\
                pdfs[galname]['bins'] ,\
                pdfs[galname]['v_avg'] = res
    if fname:
        direc = paths.data
        fname=direc+fname
        with open(fname,'wb') as f:
            pickle.dump(pdfs, f, pickle.HIGHEST_PROTOCOL)
    return pdfs

def x_3d_pdf(df, galname, bins=100, r=8.3, dr=1.5, test_unpack=None, 
             typ='fire'):
    if test_unpack:
        ms, mvir, rs, rvir, v_mags, v_vecs, parttypes = test_unpack
    else:
        ms, mvir, rs, rvir, v_mags, v_vecs, parttypes = unpack_new(df, 
                                                                   galname,
                                                                   typ=typ)[:7]
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
    ms, mvir, rs, rvir, v_mags, v_vecs, parttypes = unpack_new(df, galname)[:7]
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

def mlr(fsource, xcols, ycol, xscales=None, yscale='log', dropgals=None,
        prediction_x=None, dX=None, fore_sig=1.-0.682, beta_sig=0.05,
        verbose=False, return_band=False, return_coef_errors=False, **kwargs):
    '''
    Multi-linear regression

    dropgals: list
    '''
    assert isinstance(xcols,(list,tuple,np.ndarray))
    df=load_data(fsource)
    if dropgals is not None:
        df = df.drop(dropgals)
    if dX is not None:
        dX = np.array(dX)
    else:
        dX = np.zeros(len(xcols))
    if prediction_x is not None:
        prediction_x = np.array(prediction_x)
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
                if prediction_x is not None:
                    prediction_x[:,i] = np.log10(prediction_x[:,i])
            elif scale == 'linear':
                Xs[i]=df[xcol]
            else:
                raise ValueError('Scale should be \'log\' or \'linear\'')
        Xs=Xs.transpose()
    else:
        xscales = ['linear']*len(xcols)
        Xs=df[xcols]

    model=LinearRegression()
    model.fit(Xs, ys)
    r2=model.score(Xs, ys)
    N = len(Xs)
    k = Xs.shape[1]
    r2a = 1.-(N-1.)/(N-k-1.)*(1.-r2) #adjusted r2
    coefs=model.coef_
    intercept=model.intercept_
    ys_pred = model.predict(Xs)

    Xs=Xs.transpose()

    ###########################################################################
    # Error Analysis
    ###########################################################################
    X = Xs.transpose()
    y = np.reshape(ys.values, (len(ys),-1))
    yhat = np.reshape(ys_pred, (len(ys_pred),-1))
    # plus one because LinearRegression adds an intercept term

    #p = len(X.columns) + 1  
    p = X.shape[1] + 1  

    residuals = np.array(y - yhat)
    sse = residuals.T @ residuals
    mse = sse[0, 0] / (N - p)

    X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
    X_with_intercept[:, 0] = 1
    #X_with_intercept[:, 1:p] = X.values
    X_with_intercept[:, 1:p] = X
    #degrees of freedom
    deg = N - X_with_intercept.shape[1] 

    #beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) \
    #           @ X_with_intercept.T @ y
    beta_hat = np.concatenate(([[intercept]],
                               np.reshape(coefs,(len(coefs),-1))))
    reg_diffs = yhat - np.mean(y)
    rss = reg_diffs.T @ reg_diffs
    msr = rss[0,0] / k #mean square due to the regression

    covmat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) \
             * mse
    # standard errors of coefficients
    S = np.sqrt([[covmat[i,i]] for i in range(len(covmat))])
    T = np.abs(beta_hat) / S #t-stats for coefficients
    tc_beta = scipy.stats.t.ppf(q=1-beta_sig/2., df=deg)
    delta_beta = S*tc_beta #uncertainties in coefficients
    P = 2.*(1.-scipy.stats.t.cdf(T, deg)) #p-values
    F = msr/mse
    F_sig = 0.01 #significance
    Fc = scipy.stats.f.ppf(q=1-F_sig, dfn=k, dfd=deg)

    if verbose:
        print('Covariance:')
        print(covmat)
        print('')
        table1 = [['','coeff','+/-','t-stat', 'p-values']]
        rows = [['X_'+str(i)] for i in range(p)]
        rows = np.concatenate((rows,beta_hat,delta_beta,T,P), axis=1)
        table1 += list(rows.reshape(p,-1))
        print(tabulate.tabulate(table1, headers='firstrow', tablefmt='rst')) 
        print('t_c = {0:0.1f}'.format(tc_beta))
        print('t-test type: 2 tailed, {0:0.0f}% significance\n' \
              .format(beta_sig*100.))

        table2 = [['F','F_c','significance'],[F,Fc, F_sig]] 
        print(tabulate.tabulate(table2, headers='firstrow', tablefmt='rst'))
        print('r2 = {0:0.2f}'.format(r2))
        print('r2a = {0:0.2f}'.format(r2a))

    def calc_forecast(prediction_x, dX, verbose):
        '''
        Return both the prediction and the margin of
        error on that prediction
        '''
        prediction_x = np.array(prediction_x)
        W = np.concatenate((np.ones((prediction_x.shape[0],1),dtype=float), 
                            prediction_x),
                           axis=1)
        var_mean = np.array([[(w @ covmat) @ w.T] for w in W])
        s_mean = np.sqrt(var_mean) #std error of mean
        var_f = var_mean + mse #variance of forecast
        s_forecast = np.sqrt(var_f) #std error of forecast
        # critical 2-tailed t value  
        tc_f = scipy.stats.t.ppf(q=1.-fore_sig/2., df=N-p)
        delta_f = tc_f*s_forecast #uncertainty in the forecast

        varY_fr_X = np.empty((len(prediction_x),len(xcols)))
        # determine variance in the forecast imparted by the uncertainty in
        # X_hat (e.g. x_hat =  MW's local circular velocity):
        if scale == 'log':
            varY_fr_X =  beta_hat[1:]**2. * dX**2. \
                           / (10.**(prediction_x*2.) * np.log(10.)**2.)
        elif scale == 'linear':
            varY_fr_X = beta_hat[1:]**2. * dX**2.

        delta_f = np.sqrt(delta_f**2. + np.sum(varY_fr_X,axis=1,
                          keepdims=True))
        Y_hat = W @ beta_hat

        if verbose:
            table3 = np.concatenate((W.T, beta_hat), axis=1)
            table3 = np.concatenate(([['x_i','coeff']],table3),axis=0)
            print('')
            print(tabulate.tabulate(table3, headers='firstrow', 
                                    tablefmt='rst'))
            
            table4 = [np.concatenate([Y_hat.flatten(), [tc_f], 
                       s_forecast.flatten(), 
                       (tc_f*s_forecast).flatten(), 
                       *np.sqrt(varY_fr_X),
                       delta_f.flatten()])]
            header4 = [['yhat','t_c','std err of forecast', 't_c * std err',
                        *['dY/dX{0:d} * err(X{0:d})'.format(i+1) \
                          for i in range(k)],
                        'err(yhat)']]
            table4 = np.concatenate((header4, table4), axis=0)
            print(tabulate.tabulate(table4, headers='firstrow', 
                                    tablefmt='rst'))
            print('(t-test type: 2 tailed, {0:0.0f}% significance)\n' \
                  .format(fore_sig*100.))
         
        return Y_hat, delta_f

    ###########################################################################
    results = [coefs, intercept, r2, Xs, ys, ys_pred, r2a, residuals]
    ###########################################################################
    if return_coef_errors:
        #######################################################################
        results.append(delta_beta)
        #######################################################################
    if return_band:
        if X.shape[1:] != (1,):
            raise ValueError('I haven\'t coded this to work with multiple '
                             'regression.')
        N_uncert = 20
        X_band = np.linspace(X.min(), X.max(), N_uncert)
        X_band = X_band.reshape(N_uncert,1)
        W_band = np.concatenate((np.ones((X_band.shape[0],1),dtype=float),
                                 X_band),axis=1)
        # Assume no error in X when making the uncertainty band:
        dX_band = np.zeros((N_uncert,1)) 
        Yhat_band, delta_band = calc_forecast(X_band, dX_band, False)
        band = np.array([X_band.flatten(), 
                         (Yhat_band+delta_band).flatten(),
                         (Yhat_band-delta_band).flatten(),
                         Yhat_band.flatten()])
        #######################################################################
        results.append(band)
        #######################################################################
        
    if prediction_x is not None:
        prediction_x = np.array(prediction_x)
        W = np.concatenate((np.ones((prediction_x.shape[0],1),dtype=float), 
                            prediction_x),
                           axis=1)
        Y_hat, delta_f = calc_forecast(prediction_x, dX, verbose)
        prediction_y = [Y_hat, delta_f]
        #######################################################################
        results.append(prediction_y) 
        #######################################################################

    return tuple(results) 

def get_v_escs(fname=None, rotate=False):
    '''
    Get escape velocities for all M12's, without any pre-existing analysis
    '''

    def fill_v_escs_dic(gal):
        if rotate:
            import rotation_wrapper
            dic = rotation_wrapper.get_rotated_gal(df, gal)
        else:
            dic = unpack_4pot(df, gal)
        
        thetas = np.pi*np.array([1./4., 1./2., 3./4., 1., 5./4., 3./2., 7./4., 
                                 2.])
        v_escs[gal] = {}
        v_escs[gal]['v_escs'] = []
        
        print('Calculating v_e:')
        pbar = ProgressBar()
        for theta in pbar(thetas):
            rvec = r * np.array([np.cos(theta), np.sin(theta), 0.])
            if rotate:
                coords = dic['coord_rot']
            else:
                coords = dic['pos']
            v_escs[gal]['v_escs'] += [calc_vesc(dic['mass'], 
                                                coords, rvec).value]
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

def save_var_latex(key, value):
    dict_var = {}

    file_path = paths.paper +  "data.txt"

    try:
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass

    dict_var[key] = value

    with open(file_path, "w") as f:
        for key in dict_var.keys():
            f.write(f"{key},{dict_var[key]}\n")

    return None

def save_var_raw(dict_add):
    file_path = paths.data + 'data_raw.pkl'
    try:
        with open(file_path, 'rb') as f:
            d = pickle.load(f)
    except FileNotFoundError:
        d = {}
    d = {**d, **dict_add}

    with open(file_path, 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    return None

def save_prediction(string, y, dy):
    '''
    Save strings to data.txt to be used by paper.tex.

    Parameters
    ----------
    string: str 
	The identifier for the variable
    y: str 
        The value of the target variable
    dy: np.ndarray of str
	The uncertainty in the variable. The user *must* provide it as an
        np.ndarray, even if it's only one single symmetric uncertainty.

    Returns
    -------
    None
    ''' 

    save_var_latex(string, y)
    if len(dy)==1:
        save_var_latex('d'+string, dy[0])
        save_var_latex('d'+string+'_plus', 'n/a')
        save_var_latex('d'+string+'_minus', 'n/a')
    elif len(dy)==2:
        save_var_latex('d'+string, 'n/a')
        save_var_latex('d'+string+'_plus', dy[0])
        save_var_latex('d'+string+'_minus', dy[1])
    else:
        raise ValueError('Margin of error has a maximum of two elements.')
    return None

def save_mvir(fname, freq_save=False):
    df = init_df()
    for g in df.index:
        print('Retrieving '+g)
        analyze(df, g, dr=1.5, drsolar=None, typ='fire', 
                vcircparts=['PartType4'], source='original', dz=0.5)
        if freq_save and fname:
            save_data(df, fname)
    df_new = df[['mvir_fromhcat','mvir_stellar']]
    save_data(df, fname)
    return None

def combine_data(fname1, fname2, newfname):
    df1 = load_data(fname1)
    df2 = load_data(fname2)
    df = pd.concat([df1, df2], axis=1)
    df.attrs = df1.attrs 
    save_data(df, newfname)
    return None

def get_v0(gal):
    df = init_df()
    with open('./data/v_pdfs.pkl','rb') as f:
        pdfs_v=pickle.load(f)
    v_bins = pdfs_v[gal]['bins']
    vs = (v_bins[1:] + v_bins[:-1]) / 2.
    i = np.argmax(pdfs_v[gal]['ps'])
    v0 = vs[i]
    return v0

def save_v0s(initial_fname, new_fname):
    df = load_data(initial_fname)
    for gal in df.index:
        df.loc[gal,'v0'] = get_v0(gal) 
    save_data(df, new_fname)
    return None 
    
if __name__=='__main__':
    gen_data(fname='dm_den_20210623_2.h5')
