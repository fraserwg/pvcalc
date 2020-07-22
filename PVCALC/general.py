#!/usr/bin/env python3
""" PVCALC.general

This module contains functions which are used by two or more of the
specialised submodules.
"""
import xarray as xr

def construct_grid_file_name(processor, tile):
    ''' Gives the relative path to a grid file name

    Arguments:
        processor --> the mnc folder name (string). e.g. 'mnc_0001'
        tile --> the tile number (string). e.g. 't000002'

    Returns:
        fn --> The relative path to the grid netcdf file of the tile (sting).
            There is no guarantee that such a file exists however!
    '''
    fn = processor + '/grid.' + tile + '.nc'
    return fn


def deconstruct_processor_tile_relation(fn):
    ''' Gives the processor and tile names from a grid file name

    Arguments:
        fn --> string of the form './mnc_XXXX/VAR.tYYYY.*nc'

    Returns:
        processor --> string of the processor folder name, e.g. 'mnc_0001'
        tile --> string of the tile name, e.g. 't0001'
    '''
    processor = fn.split('/')[0]
    tile = fn.split('.')[1]
    return processor, tile


def is_boundary(da_depth):
    ''' Determines is a tile is a boundary tile

    Arguments:
        da_depth --> xarray.dataarray object. Contains depth values for
            the tile.

    Returns:
        North --> boolean, True if northern boundary tile.
        South --> boolean, True if southern boundary tile.
        East --> boolean, True if eastern boundary tile.
        West --> boolean, True if western boundary tile.

    Notes:
        The function only works if the boundary is a straight line along
        lines of lattitude or longitude. If the depth is zero at points
        adjacent to the corners of the domain the edge is deemed to be a
        boundary.
    '''
    North, South, East, West = False, False, False, False
    if not da_depth.isel({'X': 1, 'Y': -1}).data:
        North = True
    if not da_depth.isel({'X': 1, 'Y': 0}).data:
        South = True
    if not da_depth.isel({'X': -1, 'Y': 1}).data:
        East = True
    if not da_depth.isel({'X': 0, 'Y': 1}).data:
        West = True
    return North, South, East, West


def calc_q(ds_b, da_vvort, ds_hvort):
    ''' Calculates the PV by combining arrays of its constituents

    Arguments:
        ds_b --> xarray dataset containing buoyancy gradients. Must contain
            keys 'dbdz', 'dbdx' and 'dbdy'
        da_vvort --> xarray dataarray containing the vertical component of
            absolute vorticity.
        ds_hvort --> xarray dataset containing the meridional and zonal
            components of absolute vorticity. Must contain keys 'merid' and
            'zonal'

    Returns:
        q --> xarray dataset containing variable 'potVort' which gives the
            Ertel PV.

    Notes:
        All inputs must be given on the same grid.
    '''
    vert = da_vvort * ds_b['dbdz']
    merid = ds_hvort['merid'] * ds_b['dbdy']
    zonal = ds_hvort['zonal'] * ds_b['dbdx']
    q = xr.Dataset()
    q['potVort'] = vert + merid + zonal
    return q


def format_vertical_coordinates(ds, ds_grid):
    ''' adds meaningful labels and depths to a dataset output by MITgcm diags

    Arguments:
        ds --> dataset that needs the verical coordinates formatting
        ds_grid --> grid dataset containing Z, Zl and Zu coordinates which
            correspond to the formatted depths we want to add to ds

    Returns:
        ds --> dataset with meainingful vertical coordinate labels and names
            applied
    '''
    Zmd_name = 'Zmd{:06d}'.format(ds.Nr)
    Zld_name = 'Zld{:06d}'.format(ds.Nr)
    Zud_name = 'Zud{:06d}'.format(ds.Nr)

    try:
        ds.dims[Zmd_name]
        ds[Zmd_name] = ds_grid['Z'].data
        ds = ds.rename({Zmd_name: 'Z'})
    except KeyError:
        pass

    try:
        ds.dims[Zld_name]
        ds[Zld_name] = ds_grid['Zl'].data
        ds = ds.rename({Zld_name: 'Zl'})
    except KeyError:
        pass

    try:
        ds.dims[Zud_name]
        ds[Zud_name] = ds_grid['Zu'].data
        ds = ds.rename({Zud_name: 'Zu'})
    except KeyError:
        pass

    return ds
