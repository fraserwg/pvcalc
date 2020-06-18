#!/usr/bin/env python3
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


def is_boundary(depth):
    ''' Determines is a tile is a boundary tile

    Arguments:
        depth --> xarray.dataarray or dataset object. Contains depth values for
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
    if not depth.isel({'X': 1, 'Y': -1}).data:
        North = True
    if not depth.isel({'X': 1, 'Y': 0}).data:
        South = True
    if not depth.isel({'X': -1, 'Y': 1}).data:
        East = True
    if not depth.isel({'X': 0, 'Y': 1}).data:
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
