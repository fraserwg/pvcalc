#!/usr/bin/env python3
import os
import xarray as xr
import mds
from glob import glob
import warnings
from threading import Thread
from queue import Queue
import time
from multiprocessing import Pool
import general as PVG


def is_tile_in_slice(proc, tile, lat):
    grid_fn = PVG.construct_grid_file_name(proc, tile)
    ds_grid = xr.open_dataset(grid_fn)
    
    if ds_grid['Y'].isel({'Y': 1}) <= lat and ds_grid['Y'].isel({'Y': -2}) >= lat:
        tile_in_slice = True
    else:
        tile_in_slice = False
    
    return tile_in_slice, proc, tile


def _open_dataset_slice(file_list, lat, ds_grid, variable):
    ''' Opens datasets contained in file_list. Selects
        the slice corresponding to lat. Corrects
        coordinate variables. Only loads variables
        specified.
        
    Args:
        file_list --> list of file paths
        lat --> latitude of slice (float)
        ds_grid --> grid dataset corresponding to tile (xr.Dataset)
        variable --> variables to load (None, list of str or str).
            If None, all variables present loaded.
        
    Returns:
        ds_lists --> list of xarray datasets.
    
    Notes:
        Assumes that delta Y is a constant
        
    '''
    # Variable tells which data arrays within the file are wanted
    # If none is specified, all are returned
    if variable == None:
        ds_list = [xr.open_dataset(fn) for fn in file_list]
    elif type(variable) == list:
        ds = xr.open_dataset(file_list[0])
        drop_list = [elem for elem in ds.data_vars]
        [drop_list.remove(var) for var in variable]
        ds_list = [xr.open_dataset(fn).drop_vars(drop_list) for fn in file_list]
    elif type(variable) == str:
        ds = xr.open_dataset(file_list[0])
        drop_list = [elem for elem in ds.data_vars]
        drop_list.remove(variable)
        ds_list = [xr.open_dataset(fn).drop_vars(drop_list) for fn in file_list]
    else:
        raise TypeError('variable must be either str or list')
    
    # The above block gives a list of datasets corresponding to the
    # datasets in the slice.
    
    # Now let's select the data which is in the slice we want.
    dy = ds_grid.Yp1[1] - ds_grid.Yp1[0]
    try:
        ds_list = [ds.sel({'Y': [lat - dy, lat, lat + dy]},
                          method='nearest')
                   for ds in ds_list]
    except ValueError:
        pass
    
    try:
        ds_list = [ds.sel({'Yp1': [lat - dy * 0.5, lat + dy * 0.5]},
                          method='nearest')
                   for ds in ds_list]
    except ValueError:
        pass
    
    # We will go through the vertical dimensions and rename them properly
    Zmd_name = 'Zmd{:06d}'.format(ds_list[0].Nr)
    Zl_name = 'Zld{:06d}'.format(ds_list[0].Nr)
    Zu_name = 'Zud{:06d}'.format(ds_list[0].Nr)

    try:
        [ds.dims[Zmd_name] for ds in ds_list]
        ds_list2 = list()
        for ds_var in ds_list:
            ds_var[Zmd_name] = ds_grid['Z'].data
            ds_var = ds_var.rename({Zmd_name: 'Z'})
            ds_list2 += [ds_var]
        ds_list = ds_list2
    except KeyError:
        pass

    try:
        [ds.dims[Zl_name] for ds in ds_list]
        ds_list2 = list()
        for ds_var in ds_list:
            ds_var[Zl_name] = ds_grid['Zl'].data
            ds_var = ds_var.rename({Zl_name: 'Zl'})
            ds_list2 += [ds_var]
        ds_list = ds_list2
    except KeyError:
        pass

    try:
        [ds.dims[Zu_name] for ds in ds_list]
        ds_list2 = list()
        for ds_var in ds_list:
            ds_var[Zu_name] = ds_grid['Zu'].data
            ds_var = ds_var.rename({Zu_name: 'Zu'})
            ds_list2 += [ds_var]
        ds_list = ds_list2
    except KeyError:
        pass

    return ds_list


def open_tile(file, processor, tile, lat, variable=None):
    # Use glob to find all the files corresponding to the
    # tile containing relevant variables.
    search_pattern = processor + '/' + file + '*.' + tile + '.nc'
    file_list = glob(search_pattern)
    
    if len(file_list) == 0:
        raise FileNotFoundError('No files matching {} exist'.format(search_pattern))
        
    # Construct grid fn and open file
    grid_fn = PVG.construct_grid_file_name(processor, tile)
    ds_grid = xr.open_dataset(grid_fn)
    depth = ds_grid['Depth']

    # Use another function to open the slices.
    ds_list = _open_dataset_slice(file_list, lat, ds_grid, variable)

    # Join along the time axis if necessary
    if len(file_list) == 1:
        ds_var = ds_list[0]
    elif len(file_list) >= 1:
        ds_var = xr.concat([ds for ds in ds_list], dim='T')
        
    # Establish boundary points
    _, _, East, West = PVG.is_boundary(depth)
    
    if East:
        try:
            xlen = ds_var.dims['X']
            ds_var = ds_var.isel(X=slice(0, xlen - 1))
        except KeyError:
            pass

        try:
            xplen = ds_var.dims['Xp1']
            ds_var = ds_var.isel(Xp1=slice(0, xplen - 1))
        except KeyError:
            pass

    if West:
        try:
            xlen = ds_var.dims['X']
            ds_var = ds_var.isel(X=slice(1, xlen))
        except KeyError:
            pass

        try:
            xplen = ds_var.dims['Xp1']
            ds_var = ds_var.isel(Xp1=slice(1, xplen))
        except KeyError:
            pass
    
    return ds_var


def abs_vort(ds_vert, ds_grid, lat):
    da_vort = ds_vert['momVort3'] + ds_grid['fCoriG']
    da_vort = da_vort.interp({'Xp1': ds_grid['X'], 'Yp1': ds_grid['Y'].isel({'Y': 1})})
    da_vort = da_vort.drop_vars(['Xp1', 'Yp1'])
    return da_vort


def grad_b(ds_rho, rho_ref):
    g = 9.81  # m / s^2
    rho_0 = 1000  # kg / m^3

    ds_b = xr.Dataset()
    b = - g * (ds_rho['RHOAnoma'] + rho_ref) / rho_0

    ds_b['dbdz'] = b.isel({'Y': 1}).differentiate('Z', edge_order=2)
    ds_b['dbdx'] = b.isel({'Y': 1}).differentiate('X', edge_order=2)
    ds_b['dbdy'] = b.differentiate('Y').isel({'Y': 1})
    return ds_b


def hor_vort(ds_vel, fCoriCos):
    ds_hv = xr.Dataset()
    ds_hv['dUdZ'] = ds_vel['UVEL'].isel({'Y': 1}).differentiate(
        'Z', edge_order=1).interp({'Xp1': ds_vel['X']})

    ds_hv['dVdZ'] = ds_vel['VVEL'].differentiate(
        'Z', edge_order=1).interp({'Yp1': ds_vel['Y'].isel({'Y': 1})})
    
    ds_hv['dWdX'] = ds_vel['WVEL'].isel({'Y': 1}).differentiate('X', edge_order=2).interp({'Zl': ds_vel['Z']})

    ds_hv['dWdY'] = ds_vel['WVEL'].differentiate('Y', edge_order=1).isel({'Y': 1}).interp({'Zl': ds_vel['Z']})

    ds_vort = xr.Dataset()
    ds_vort['merid'] = ds_hv['dUdZ'] - ds_hv['dWdX'] + fCoriCos
    ds_vort['zonal'] = ds_hv['dWdY'] - ds_hv['dVdZ']
    return ds_vort


def calc_pv_of_tile(proc_tile, lat, fCoriCos):
    ds_vel = open_tile('Velocity', *proc_tile, lat)
    ds_grid = open_tile('grid', *proc_tile, lat)
    ds_vert = open_tile('Vorticity', *proc_tile, lat)
    ds_rho = open_tile('Rho', *proc_tile, lat)
    rho_ref = mds.rdmds('RhoRef')
    
    # This bit needs some thread based parallelism
    da_vvort = abs_vort(ds_vert, ds_grid, lat)
    ds_b = grad_b(ds_rho, rho_ref)
    ds_hvort = hor_vort(ds_vel, fCoriCos)
    
    q = PVG.calc_q(ds_b, da_vvort, ds_hvort)
    q.attrs = ds_vel.attrs
    
    North, South, East, West = PVG.is_boundary(ds_grid['Depth'])
    if not East:
        q = q.isel({'X': slice(0, -1)})
    if not West:
        xid = q.dims['X']
        q = q.isel({'X': slice(1, xid)})
    return q