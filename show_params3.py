#!/usr/bin/env python2

import h5py
import sys

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
