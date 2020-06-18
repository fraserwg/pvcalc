#!/usr/bin/env python3
import time
import os
import xarray as xr
import mds
from glob import glob
import warnings
from threading import Thread
from queue import Queue
from multiprocessing import Pool
import getopt
import sys
from . import general as PVG


def _open_dataset_level(file_list, lvl, ds_grid, variable):
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

    Zmd_name = 'Zmd{:06d}'.format(ds_list[0].Nr)
    Zl_name = 'Zld{:06d}'.format(ds_list[0].Nr)
    Zu_name = 'Zud{:06d}'.format(ds_list[0].Nr)

    try:
        [ds.dims[Zmd_name] for ds in ds_list]
        ds_list = [ds.isel({Zmd_name: slice(lvl - 1, lvl + 2)})
                   for ds in ds_list]
        ds_list2 = list()
        for ds_var in ds_list:
            ds_var[Zmd_name] = ds_grid['Z'].data[slice(lvl - 1, lvl + 2)]
            ds_var = ds_var.rename({Zmd_name: 'Z'})
            ds_list2 += [ds_var]
        ds_list = ds_list2
    except KeyError:
        pass

    try:
        [ds.dims[Zl_name] for ds in ds_list]
        ds_list = [ds.isel({Zl_name: slice(lvl, lvl + 2)}) for ds in ds_list]
        ds_list2 = list()
        for ds_var in ds_list:
            ds_var[Zl_name] = ds_grid['Zl'].data[slice(lvl, lvl + 2)]
            ds_var = ds_var.rename({Zl_name: 'Zl'})
            ds_list2 += [ds_var]
        ds_list = ds_list2
    except KeyError:
        pass

    try:
        [ds.dims[Zu_name] for ds in ds_list]
        ds_list = [ds.isel({Zu_name: slice(lvl - 1, lvl + 1)})
                   for ds in ds_list]
        ds_var[Zu_name] = ds_grid['Zu'].data[slice(lvl - 1, lvl + 1)]
        ds_var = ds_var.rename({Zu_name: 'Zu'})

        [ds.dims[Zu_name] for ds in ds_list]
        ds_list = [ds.isel({Zu_name: slice(lvl - 1, lvl + 1)})
                   for ds in ds_list]
        ds_list2 = list()
        for ds_var in ds_list:
            ds_var[Zu_name] = ds_grid['Zu'].data[slice(lvl - 1, lvl + 1)]
            ds_var = ds_var.rename({Zu_name: 'Zu'})
            ds_list2 += [ds_var]
        ds_list = ds_list2
    except KeyError:
        pass

    return ds_list


def open_tile(file, processor, tile, lvl=1, variable=None):
    # Construct the search pattern and list the associated files
    if file == 'grid':
        file_name = processor + '/' + file + '.' + tile + '.nc'
        file_list = glob(file_name)
    else:
        search_pattern = processor + '/' + file + '.*.' + tile + '.nc'
        file_list = glob(search_pattern)

    if len(file_list) == 0:
        raise FileNotFoundError(
            'No files matching {} exist'.format(search_pattern))

    # Construct grid fn and open file
    grid_fn = PVG.construct_grid_file_name(processor, tile)
    ds_grid = xr.open_dataset(grid_fn)
    depth = ds_grid['Depth']

    ds_list = _open_dataset_level(file_list, lvl, ds_grid, variable)
    if len(file_list) == 1:
        # No temporal joining necessary
        ds_var = ds_list[0]
    elif len(file_list) >= 1:
        ds_var = xr.concat([ds for ds in ds_list], dim='T')

    # Establish boundary points
    North, South, East, West = PVG.is_boundary(depth)

    # Remove boundary points if necessary
    if North:
        try:
            ylen = ds_var.dims['Y']
            ds_var = ds_var.isel(Y=slice(0, ylen - 1))
        except KeyError:
            pass

        try:
            yplen = ds_var.dims['Yp1']
            ds_var = ds_var.isel(Yp1=slice(0, yplen - 1))
        except KeyError:
            pass

    if South:
        try:
            ylen = ds_var.dims['Y']
            ds_var = ds_var.isel(Y=slice(1, ylen))
        except KeyError:
            pass

        try:
            yplen = ds_var.dims['Yp1']
            ds_var = ds_var.isel(Yp1=slice(1, yplen))
        except KeyError:
            pass

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


def grad_b(ds_rho, rho_ref):
    ''' Calculates the gradient of the buoyancy field from the density field.

    Arguments:
        ds_rho --> xarray dataset containing 'RHOAnoma', the density anomaly.
            Should be opened using open_tile
        rho_ref --> The reference density which added to the dat in RHOAnoma
            gives the total density. Should be opened using mds then sliced.

    Returns:
        ds_b --> dataset containing the buoyancy gradients and buoyancy.

    Notes:
        rho_ref is NOT the density of the reference level. That is set to
            1000 kg/m^3 in the below function.
    '''
    g = 9.81  # m / s^2
    rho_0 = 1000  # kg / m^3

    ds_b = xr.Dataset()
    ds_b['b'] = - g * (ds_rho['RHOAnoma'] + rho_ref) / rho_0

    ds_b['dbdz'] = ds_b['b'].differentiate('Z', edge_order=2).isel({'Z': 1})
    ds_b['dbdx'] = ds_b['b'].isel({'Z': 1}).differentiate('X', edge_order=2)
    ds_b['dbdy'] = ds_b['b'].isel({'Z': 1}).differentiate('Y', edge_order=2)
    #ds_b = ds_b.isel({'Z': slice(0, -1)})
    return ds_b


def hor_vort(ds_vel, fCoriCos):
    ''' Calculates the horizontal vorticity.

    Arguments:
        ds_vel --> xarray dataset of velocities as opened by open_tile. Should
            contain the ketys 'UVEL', 'VVEL' and 'WVEL' defined on the C-grid.
        fCoriCos --> Non-traditional component of the Coriolis force. Either a
            float or an xarray dataarray or numpy array defined on the 'X' and
            'Y' points of the tile.

    Returns:
        ds_vort --> xarray dataset containing the horizontal components of the
            absolute vorticity.
    '''
    ds_hv = xr.Dataset()
    ds_hv['dUdZ'] = ds_vel['UVEL'].differentiate(
        'Z', edge_order=1).interp({'Xp1': ds_vel['X']}).isel({'Z': 1})
    ds_hv['dVdZ'] = ds_vel['VVEL'].differentiate(
        'Z', edge_order=1).interp({'Yp1': ds_vel['Y']}).isel({'Z': 1})
    ds_hv['dWdX'] = ds_vel['WVEL'].interp(
        {'Zl': ds_vel['Z'].isel({'Z': 1})}).differentiate('X', edge_order=2)
    ds_hv['dWdY'] = ds_vel['WVEL'].interp(
        {'Zl': ds_vel['Z'].isel({'Z': 1})}).differentiate('Y', edge_order=2)

    ds_vort = xr.Dataset()
    ds_vort['merid'] = ds_hv['dUdZ'] - ds_hv['dWdX'] + fCoriCos
    ds_vort['zonal'] = ds_hv['dWdY'] - ds_hv['dVdZ']
    return ds_vort


def abs_vort(ds_vert, ds_grid):
    ''' Calculates the vertical component of the absolute vorticity.

    Arguments:
        ds_vert --> xarray dataset containing the vertical component of the
            relative vorticity. Contains the key 'momVort3'.
        ds_grid --> xarray dataset containing the grid data of the domain. Must
            contain the key 'fCoriG' which is the vertical component of the
            planetary vorticity defined at the cell corners.

    Returns:
        da_vort --> xarray dataarray containing the vertical component of the
            absolute vorticity.
    '''
    ds_vert = ds_vert.isel({'Z': 1})
    ds_vort = xr.DataArray()
    da_vort = ds_vert['momVort3'] + ds_grid['fCoriG']
    da_vort = da_vort.interp({'Xp1': ds_grid['X'], 'Yp1': ds_grid['Y']})
    da_vort = da_vort.drop_vars(['Xp1', 'Yp1'])
    return da_vort


def calc_pv_of_tile(proc_tile, lvl, fCoriCos):
    pt = proc_tile
    ds_rho = open_tile('Rho', *pt, lvl=lvl)
    rho_ref = mds.rdmds('RhoRef')[slice(lvl - 1, lvl + 2)]
    ds_vert = open_tile('Vorticity', *pt, lvl=lvl)
    ds_grid = open_tile('grid', *pt, lvl=lvl)
    ds_vel = open_tile('Velocity', *pt, lvl=lvl)

    # Calculate vorticity and buoyancy gradients in parallel (thread based).
    que = Queue()

    b_thread = Thread(target=lambda q, ds_rho, rho_ref: q.put(
        grad_b(ds_rho, rho_ref)), args=(que, ds_rho, rho_ref))
    vv_thread = Thread(target=lambda q, ds_vert, ds_grid: q.put(
        abs_vort(ds_vert, ds_grid)), args=(que, ds_vert, ds_grid))
    hv_thread = Thread(target=lambda q, ds_vel, fCoriCos: q.put(
        hor_vort(ds_vel, fCoriCos)), args=(que, ds_vel, fCoriCos))

    b_thread.start()
    vv_thread.start()
    hv_thread.start()

    b_thread.join()
    vv_thread.join()
    hv_thread.join()

    # This is horribly hacky but works
    while not que.empty():
        result = que.get()
        if type(result) == xr.DataArray:
            da_vvort = result
        else:
            try:
                result['dbdz']
                ds_b = result
            except KeyError:
                ds_hvort = result

    # Calculate q
    q = PVG.calc_q(ds_b, da_vvort, ds_hvort)
    q.attrs = ds_vel.attrs

    # Now we need to get rid of incorrect overlap points.
    # Note that this is different to the previous cropping as that
    # was to remove land points.
    q = remove_overlap_points(q, ds_grid['Depth'])

    # Save to a netcdf file
    proc, tile = pt
    outname = './' + proc + '/PV.' + tile + '.nc'
    q.to_netcdf(outname)
    return q


def remove_overlap_points(ds, depth):
    ''' Removes points from the tile which aren't boundary points and so
    overlap with other tiles. Quantities along these edges are incorrect.

    Arguments:
        ds --> xarray dataset whose edge points we wish to remove
        depth --> xarray dataarray containing the depth of the tile.

    Returns:
        ds --> xarray dataset, cropped to remove incorrect overlap points.
    '''
    North, South, East, West = PVG.is_boundary(depth)
    if not North:
        ds = ds.isel({'Y': slice(0, -1)})
    if not South:
        yid = ds.dims['Y']
        ds = ds.isel({'Y': slice(1, yid)})
    if not East:
        ds = ds.isel({'X': slice(0, -1)})
    if not West:
        xid = ds.dims['X']
        ds = ds.isel({'X': slice(1, xid)})
    return ds
