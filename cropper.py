import h5py
import os
import rotate_galaxy
import numpy as np
from progressbar import ProgressBar

def gen_gal_data(galname):
    '''
    Generate cropped data from the original hdf5 files. Data is cropped at a
    certain radius from the center of the galaxy
    '''

    import dm_den

    print('Generating {0:s} data'.format(galname))

    min_radius = 0. #kpc
    max_radius = 10. #kpc

    df = dm_den.init_df()

    suffix=df.loc[galname,'fsuffix']
    suffix_cropped = df.loc[galname, 'fsuffix_cropped']
    res=df.loc[galname,'res']
    host_key=df.loc[galname,'host_key']
    mass_class = df.loc[galname,'mass_class']
    typ = 'fire'

    #original directiory result:
    orig_dir_res = dm_den.build_direcs(suffix, res, mass_class, typ,
                                       source='original')
    halodirec, snapdir_orig, almost_full_path_orig, num_files = orig_dir_res
    #cropped directory result:
    crop_dir_res = dm_den.build_direcs(suffix_cropped, res, mass_class, typ,
                                       source='cropped',
                                       min_radius=0.,
                                       max_radius=10.)
    _, snapdir_crop, almost_full_path_crop, _ = crop_dir_res

    if not os.path.isdir(snapdir_crop):
        os.makedirs(snapdir_crop)
    with h5py.File(almost_full_path_crop+'.hdf5',
                   'w') as f_crop:
        d = {}
        pbar=ProgressBar()
        for i in pbar(range(0,num_files)):
            with h5py.File(almost_full_path_orig+'.'+str(i)+'.hdf5', 'r') as f:
                for key1 in f.keys():
                    if key1=='Header':
                        #Just directly copy f['Header'] into f_crop.
                        #It should be the same for every snapshot subfile, so 
                        #we
                        #don't need to change it into a manageable object.
                        if key1 in f_crop.keys():
                            #We already have the header from the last snapshot
                            #subfile.
                            continue #move onto the next key1
                        f.copy(key1,f_crop)
                        continue #move onto the next key1
                    if key1 not in d:
                        #If this is the first run, initialize the 
                        #sub-dictionary
                        #corresponding to key1:
                        d[key1]=dict()
                    for key2 in f[key1].keys():
                        #Extract the data from the `h5py._hl.group.Group`s
                        new_data = f[key1][key2][:]
                        # Getting rid of the following 2 lines that save masses
                        # as float128's because it causes big problems with
                        # MacOS.
                        #if key2 in ['Masses']:
                        #    new_data = new_data.astype(np.longdouble)
                        if key2 in d[key1]:
                            #If this isn't the first snapshot subfile, add the 
                            #new
                            #data to the data as of the last subfile.
                            d[key1][key2] = np.concatenate((d[key1][key2], 
                                                            new_data))
                        else:
                            d[key1][key2] = new_data

        # Getting host halo info
        center_coord, rvir, v_halo, mvir = dm_den.get_halo_info(halodirec,
                                                                suffix, 
                                                                typ, 
                                                                host_key, 
                                                                mass_class)
        h = f_crop['Header'].attrs['HubbleParam']

        for key1 in d.keys():
            d[key1]['coord_phys'] = d[key1]['Coordinates']/h
            d[key1]['coord_centered'] = d[key1]['coord_phys']-center_coord
            d[key1]['v_vec_centered'] = d[key1]['Velocities']-v_halo
            d[key1]['r'] = np.linalg.norm(d[key1]['coord_centered'], axis=1)
            d[key1]['mass_phys'] = d[key1]['Masses']/h #units of 1e10 M_sun

            within_crop = (d[key1]['r'] <= max_radius) \
                          & (d[key1]['r'] >= min_radius)

            for key2 in d[key1].keys():
                d[key1][key2] = d[key1][key2][within_crop] #crop the dictionary
        
        #calculate temperatures
        he_fracs = d['PartType0']['Metallicity'][:,1]
        d['PartType0']['T'] = dm_den.calc_temps(he_fracs,
                                                *[d['PartType0'][key] \
                                                  for key in \
                                                  ['ElectronAbundance',
                                                   'InternalEnergy']])

        rotation_matrix = rotate_galaxy.rotation_matrix_fr_dat(
                *[flatten_particle_data(d, data) for data in ['coord_centered',
                                                              'v_vec_centered',
                                                              'mass_phys',
                                                              'r']])
        for ptcl in d.keys(): #for each particle type
            print('Rotating {0:s}'.format(ptcl))
            d[ptcl]['v_vec_rot'] = rotate_galaxy.rotate(
                    d[ptcl]['v_vec_centered'], rotation_matrix)
            d[ptcl]['coord_rot'] = rotate_galaxy.rotate(
                    d[ptcl]['coord_centered'], rotation_matrix)

            #xy component of velocity
            d[ptcl]['v_vec_disc'] = d[ptcl]['v_vec_rot'][:,:2] 
            #xy component of coordinates
            d[ptcl]['coord_disc'] = d[ptcl]['coord_rot'][:,:2] 

            ###################################################################
            ## Find the projection of velocity onto the xy vector (i.e. v_r) 
            ## v_r = v dot r / r^2 * r 
            ###################################################################
            vdotrs = np.sum(d[ptcl]['v_vec_disc'] \
                           * d[ptcl]['coord_disc'], axis=1)
            rmags = np.linalg.norm(d[ptcl]['coord_disc'], axis=1)
            d[ptcl]['v_dot_rhat'] = vdotrs / rmags

            #v dot r / r^2
            vdotrs_r2 = (vdotrs \
                / np.linalg.norm(d[ptcl]['coord_disc'], axis=1) **2.)
            #Need to reshape v dot r / r^2 so its shape=(number of particles,1)
            #so we can mutilpy those scalars by the r vector
            vdotrs_r2 = vdotrs_r2.reshape(len(d[ptcl]['coord_disc']), 1)
            d[ptcl]['v_r_vec'] = vdotrs_r2 * d[ptcl]['coord_disc']
            ###################################################################

            #v_phi is the difference of the xy velocity and v_r
            d[ptcl]['v_phi_vec'] = d[ptcl]['v_vec_disc'] - d[ptcl]['v_r_vec']
            d[ptcl]['v_phi_mag'] = np.linalg.norm(d[ptcl]['v_phi_vec'],axis=1)

            ###################################################################
            ## Finding the vphi in \vec{vphi} = vphi * \hat{vphi} 
            ## (i.e. v dot phi )
            ## Note v_phi_mag = |v dot phi|, and we want to determine whether
            ## v dot phi is positive or negative.
            ###################################################################
            rxvs = np.cross(d[ptcl]['coord_disc'], 
                            d[ptcl]['v_vec_disc']) #r cross v
            # v dot phi is positive if the z component of r cross v is 
            # positive,
            # because this means its angular momentum is in the same direction
            # as the disc's.
            signs = rxvs/np.abs(rxvs) 
            d[ptcl]['v_dot_phihat'] = d[ptcl]['v_phi_mag']*signs
            ###################################################################

            d[ptcl]['v_dot_zhat'] = d[ptcl]['v_vec_rot'][:,2]

            l = len(d[ptcl]['mass_phys'])
            for key in d[ptcl].keys():
                try:
                    assert len(d[ptcl][key])==l
                except:
                    print('failed on '+key)

        for ptcl in d.keys(): #for each particle type
            f_crop.create_group(ptcl)
            for key2 in d[ptcl].keys():
                print('saving {0:s}, {1:s}'.format(ptcl,key2))
                data = d[ptcl][key2]
                f_crop[ptcl].create_dataset(key2, data=data, dtype=data.dtype)
        print('')

    return d 

def gen_all_gals_data():
    from dm_den import init_df
    
    df = init_df()
    for gal in df.index:
        gen_gal_data(gal) 
    
    return None

def flatten_particle_data(d, data, drop_particles=['PartType2']):
    '''
    Given a galaxy dictionary layered by particle in the same way as the 
    original hdf5
    files, return data from that dictionary for all particle types,
    marginalizing particle type information.

    Parameters:
        d: dict 
            Galaxy dictionary
        data: str
            The key corresponding to the data the user wants to
            extract
        drop_particles: list of str
            The particle types that should not be included in the flattened
            data

            'PartType0' is gas. 'PartType1' is dark matter.
            'PartType2' is dummy collisionless. 'PartType3' is grains/PIC
            particles. 'PartType4' is stars. 'PartType5' is black holes / 
            sinks.
    '''
    
    keys = list(d.keys())
    try:
        keys.remove('Header')
    except:
        pass
    for part in drop_particles:
        keys.remove(part)
    data_flat = np.concatenate([d[parttype][data] for parttype in keys])

    return data_flat

def load_data(galname, getparts='all', verbose=True):
    '''
    Load data from the new cropped hdf5 files, as opposed to the way we used to
    load data from the original hdf5 files.

    Parameters
    ----------
    galname: str
        The galaxy name string corresponding to an index in df.
    getparts: list of str: {'PartType0' : 'PartType4'}
        Specifies the particle types to extract
    v: bool
        If True, the function prints which galaxy its pulling with a progress
        bar.

    Returns
    -------
    d: dict
        Galaxy dictionary split by particle type
    '''

    if verbose:
        print('Loading {0:s}'.format(galname))

    from dm_den import init_df as init_df
    from dm_den import build_direcs 

    min_radius = 0. #kpc
    max_radius = 10. #kpc

    df = init_df()

    print(galname)
    suffix=df.loc[galname,'fsuffix']
    suffix_cropped = df.loc[galname, 'fsuffix_cropped']
    res=df.loc[galname,'res']
    host_key=df.loc[galname,'host_key']
    mass_class = df.loc[galname,'mass_class']
    typ = 'fire'

    #cropped directory result
    crop_dir_res = build_direcs(suffix_cropped, res, mass_class, typ,
                                   source='cropped',
                                   min_radius=0.,
                                   max_radius=10.)
    _, snapdir_crop, almost_full_path_crop, _ = crop_dir_res

    with h5py.File(almost_full_path_crop+'.hdf5',
                   'r') as f:
        class mydict(dict):
            # Need this so I can add attributes to the dictionary
            pass
        d = mydict()
        d.attrs = {}
        for attr in f['Header'].attrs.keys():
            d.attrs[attr] = f['Header'].attrs[attr]
        
        if getparts=='all':
            key1s = list(f.keys())
            key1s.remove('Header')
        else: 
            key1s = getparts
        if verbose:
            pbar = ProgressBar()
        else:
            #pbar wrapper does nothing
            pbar = lambda function: function
        for ptcl in pbar(key1s):
            d[ptcl] = {}
            key2s = f[ptcl].keys()
            for datapnt in key2s:
                d[ptcl][datapnt] = f[ptcl][datapnt][:]
    return d

if __name__ == '__main__':
    gen_all_gals_data()
