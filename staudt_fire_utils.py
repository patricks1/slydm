import h5py
import sys
import numpy as np
from progressbar import ProgressBar

def get_data(filename,num_of_file,key1,key2):
    if num_of_file == 1:
        f = h5py.File(filename+'.hdf5', 'r')
        if key1 == 'Header':
            return f[key1].attrs[key2]
        else:
            return f[key1][key2][:]
    else:
        pbar=ProgressBar()
        for i in pbar(range(0,num_of_file)):
            f = h5py.File(filename+'.'+str(i)+'.hdf5', 'r')
            if key1 == 'Header':
                return f[key1].attrs[key2]
            else:
                if ( len(f[key1][key2][:].shape)==1 ):
                    if i==0:
                        result = f[key1][key2][:]
                    else:
                        result = np.hstack( (result,f[key1][key2][:]) )
                else:
                    if i==0:
                        result = f[key1][key2][:]
                    else:
                        result = np.vstack( (result,f[key1][key2][:]) )
        return result

def show(f):
    snapshot = h5py.File(f, 'r')

    print(list(snapshot.keys()))
    print('====================')
    print('')
    print('redshift:', snapshot['Header'].attrs['Redshift'])
    print('box size:', snapshot['Header'].attrs['BoxSize'], 'kpc/h or Mpc/h')  
    print('mass table:', snapshot['Header'].attrs['MassTable'])
    print('Omega0:', snapshot['Header'].attrs['Omega0']) 
    print('OmegaLambda:', snapshot['Header'].attrs['OmegaLambda'])
    print('HubbleParam:', snapshot['Header'].attrs['HubbleParam'])   
    print('====================')
    print('')
    z = snapshot['Header'].attrs['Redshift'] 
    a = 1./(z+1.)
    print('Hi-res particles')
    print('====================')
    print(snapshot['PartType1']['Masses'][:])
    print(a * snapshot['PartType1']['Coordinates'][:])
    
    return None

def load_disp_vc(fname):
    direc='/export/nfs0home/pstaudt/projects/project01/data/'
    fname=direc+fname
    with h5py.File(fname,'r') as f:
        rs=np.array(f.get('rs'))
        disps_fire=np.array(f.get('disps_fire'))
        vcs_fire=np.array(f.get('vcircs_fire'))
        disps_dmo=np.array(f.get('disps_dmo'))
        vcs_dmo=np.array(f.get('vcircs_dmo'))
    return rs, disps_fire, vcs_fire, disps_dmo, vcs_dmo
