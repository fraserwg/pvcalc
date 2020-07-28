#!/usr/bin/env python3
""" slice_.py

Contains functions for calculating the PV of a slice at a constant
lattitude.
"""
from glob import glob
import xarray as xr
import MITgcmutils.mds as mds
from . import general as PVG


def is_tile_in_slice(proc, tile, lat):
    """ Determines whether a given tile is in the slice of interest

    Arguments:
        proc (str) --> Processor folder, e.g. 'mnc_0001'.
        tile (str) --> Tile file suffix, e.g. 't003'.
        lat (float) --> Latitude of slice in m.
    """
    grid_fn = PVG.construct_grid_file_name(proc, tile)
    ds_grid = xr.open_dataset(grid_fn)

    if ds_grid['Y'].isel({'Y': 1}) <= lat and ds_grid['Y'].isel({'Y': -2}) >= lat:
        tile_in_slice = True
    else:
        tile_in_slice = False

    return tile_in_slice


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
    if variable is None:
        ds_list = [xr.open_dataset(fn) for fn in file_list]
    elif isinstance(variable, list):
        ds = xr.open_dataset(file_list[0])
        drop_list = [elem for elem in ds.data_vars]
        [drop_list.remove(var) for var in variable]
        ds_list = [xr.open_dataset(fn).drop_vars(drop_list) for fn in file_list]
    elif isinstance(variable, str):
        ds = xr.open_dataset(file_list[0])
        drop_list = [elem for elem in ds.data_vars]
        drop_list.remove(variable)
        ds_list = [xr.open_dataset(fn).drop_vars(drop_list) for fn in file_list]
    else:
        raise TypeError('variable must be either str or list')

    # The above block gives a list of datasets corresponding to the
    # datasets in the slice. This should remain here.

    # Now let's select the data which is in the slice we want.
    # This should be a sub-function
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

    # Go through the vertical dimensions and rename them properly
    ds_list = [PVG.format_vertical_coordinates(ds, ds_grid) for ds in ds_list]
    return ds_list


def open_tile(file, processor, tile, lat, variable=None):
    """ Opens all the desired files corresponding to particular tile

    Arguments:
        file (str) --> File prefix, e.g. 'Vorticity'.
        processor (str) --> Folder of processor, e.g. 'mnc_0001'.
        tile (str) --> Tile file suffix, e.g. 't003'.
        lat (float) --> Latitude of slice in m.
        variable (list or None) --> List of variables to load, e.g.
            ['momVort3'].

    Returns:
        ds_var (xarray.Dataset) --> ds containing desired variable, with some
            preprocessing performed.
    """
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
    # The below should be put into PVG
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


def abs_vort(ds_vert, ds_grid):
    """ Calculates the locally vertical absolute vorticity

    Arguments:
        ds_vert (xarray.Dataset) --> dataset containing 'momVort3'.
        ds_grid (xarray.Dataset) --> dataset containing grid info.

    Returns:
        da_vort (xarray.DataArray) --> DataArray containing the absolute
            vorticity.
    """
    da_vort = ds_vert['momVort3'] + ds_grid['fCoriG']
    da_vort = da_vort.interp({'Xp1': ds_grid['X'], 'Yp1': ds_grid['Y'].isel({'Y': 1})})
    da_vort = da_vort.drop_vars(['Xp1', 'Yp1'])
    return da_vort


def grad_b(ds_rho, rho_ref):
    """ Calculates teh gradient of the buoyancy field.

    Arguments:
        ds_rho (xarray.Dataset) --> Dataset containing 'RHOAnoma'
        rho_ref (numpy.array) --> Array of reference density at each model
            level.

    Returns:
        ds_b (xarray.Dataset) --> Dataset containing the gradient of the
            buoyancy field.
    """
    g = 9.81  # m / s^2
    rho_0 = 1000  # kg / m^3

    ds_b = xr.Dataset()
    b = - g * (ds_rho['RHOAnoma'] + rho_ref) / rho_0

    ds_b['dbdz'] = b.isel({'Y': 1}).differentiate('Z', edge_order=2)
    ds_b['dbdx'] = b.isel({'Y': 1}).differentiate('X', edge_order=2)
    ds_b['dbdy'] = b.differentiate('Y').isel({'Y': 1})
    return ds_b


def hor_vort(ds_vel, fCoriCos):
    """ Calculates the horizontal components of vorticity

    Arguments:
        ds_vel (xarray.Dataset) --> Dataset containing 'UVEL', 'VVEL' and
            'WVEL'.
        fCoriCos (float or numpy.array or xarray.DataArray) --> Non-traditional
            (i.e. meridional) component of the Coriolis parameter.

    Returns:
        ds_vort (xarray.Dataset)--> Absoulte vorticity about the zonal and
            meridional directions.
    """
    ds_hv = xr.Dataset()
    ds_hv['dUdZ'] = ds_vel['UVEL'].isel({'Y': 1}).differentiate(
        'Z', edge_order=1).interp({'Xp1': ds_vel['X']})

    ds_hv['dVdZ'] = ds_vel['VVEL'].differentiate(
        'Z', edge_order=1).interp({'Yp1': ds_vel['Y'].isel({'Y': 1})})

    ds_hv['dWdX'] = ds_vel['WVEL'].isel({'Y': 1}).differentiate('X', edge_order=2).interp({'Zl': ds_vel['Z']})

    ds_hv['dWdY'] = ds_vel['WVEL'].differentiate('Y',edge_order=1).isel({'Y': 1}).interp({'Zl': ds_vel['Z']})

    ds_vort = xr.Dataset()
    ds_vort['merid'] = ds_hv['dUdZ'] - ds_hv['dWdX'] + fCoriCos
    ds_vort['zonal'] = ds_hv['dWdY'] - ds_hv['dVdZ']
    return ds_vort


def calc_pv_of_tile(proc_tile, lat, fCoriCos):
    """ Function to calculate the PV of the tile and level.
    Arguments:
        proc_tile (tuple) --> tuple of processor and tile strings, e.g.
            ('mnc_0001', 't003')
        lat (float) --> Latitude at which to calculate PV in m.
        fCoriCos (float) --> Non-traditional component of the Coriolis force.

    Returns:
        q (xarray.Dataset) --> dataset containing PV of the tile

    Notes:
        - The function opens all the required files and calculates the
            quantities required to calculate PV using threads.
        - It then retrieves the output of each thread and combines the output
            to calculate the PV.
        - Metadata is added to the resulting dataset. The data is then cleaned.
        - The tile's PV is then saved as an intermediate netCDF file.
    """
    ds_vel = open_tile('Velocity', *proc_tile, lat)
    ds_grid = open_tile('grid', *proc_tile, lat)
    ds_vert = open_tile('Vorticity', *proc_tile, lat)
    ds_rho = open_tile('Rho', *proc_tile, lat)
    rho_ref = mds.rdmds('RhoRef')

    # This bit needs some thread based parallelism
    da_vvort = abs_vort(ds_vert, ds_grid)
    ds_b = grad_b(ds_rho, rho_ref)
    ds_hvort = hor_vort(ds_vel, fCoriCos)

    q = PVG.calc_q(ds_b, da_vvort, ds_hvort)
    q.attrs = ds_vel.attrs

    _, _, East, West = PVG.is_boundary(ds_grid['Depth'])
    if not East:
        q = q.isel({'X': slice(0, -1)})
    if not West:
        xid = q.dims['X']
        q = q.isel({'X': slice(1, xid)})
    return q

def approximate_overturning_streamfunction(proc_tile, lat):
    # convention: positive = anticlockwise
    dsvel = open_tile('Velocity', *proc_tile, lat, variable='WVEL')
    psi = -dsvel.WVEL[:, ::-1].cumsum('X')[:, ::-1] * 2e3
    psi = psi.to_dataset(name='psi')
    psi.attrs = {'long_name': 'overturning stream function',
                 'calculated_from': 'WVEL',
                 'sign_convention': 'positive = anticlockwise',
                 'units': 'm^2 s^{-1}'}
    return psi
