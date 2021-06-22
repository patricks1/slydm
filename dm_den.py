import h5py
import os
import numpy as np
from staudt_fire_utils import get_data, show

def build_direcs(gal, res):
    res=str(res)
    hdirec='/data17/grenache/aalazar/FIRE/GVB/m12'+gal+'_res'+res+\
          '/halo/rockstar_dm/hdf5/halo_600.hdf5'
    direc='/data17/grenache/aalazar/FIRE/GVB/m12'+gal+'_res'+res+\
          '/output/hdf5/'
    return hdirec, direc


def analyze(gal, res, dr=0.1, host_key='host.index'):
    halodirec, direc = build_direcs(gal, res)
    s=600
    s=str(s)
    sdir=direc+'snapdir_'+s+'/snapshot_'+s

    #Determine number of files
    N=len(os.listdir(direc+'snapdir_'+s+'/'))

    coords=get_data(sdir,N,'PartType1','Coordinates')
    h=get_data(sdir,2,'Header','HubbleParam')
    ms=get_data(sdir,N,'PartType1','Masses')*10.**10./h
    vs=get_data(sdir,N,'PartType1','Velocities')
    coords/=h
    vs/=h

    # Getting host halo info
    with h5py.File(halodirec,'r') as f:
        if gal=='_elvis_RomeoJuliet' and host_key=='host2.index':
            #RomeoJuliet doesn't have a halo2.index key
            ms_hals=f['mass.vir'][:]
            is_sorted=np.argsort(ms_hals)
            i=is_sorted[-2]
        else:
            i=f[host_key][0]
        p=f['position'][i]
        r=f['radius'][i] #Is this virial radius?
        v=f['velocity'][i]
        mvir_fromhcat=f['mass.vir'][i]

    rs=np.linalg.norm(coords-p,axis=1)
    vs=np.linalg.norm(vs-v,axis=1)

    rsun=8.3

    def get_den_disp(r):
        rmax=r+dr/2.
        rmin=r-dr/2.
        is_in=(rs<rmax) & (rs>rmin)
        v=4./3.*np.pi*(rmax**3.-rmin**3.) #kpc^3
        mtot=ms[is_in].sum()
        den=mtot/v
        disp=np.std(vs[is_in])
        return den, disp

    def get_mwithin(r):
        is_in=rs<r
        mtot=ms[is_in].sum()
        return mtot

    den_lcl, disp_lcl = get_den_disp(rsun)
    den_vir, disp_vir = get_den_disp(r)
    mvir_fromr = get_mwithin(r)

    return den_lcl, den_vir, disp_lcl, disp_vir, mvir_fromhcat, mvir_fromr
