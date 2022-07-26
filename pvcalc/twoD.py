""" PVCALC.twoD

Functions for calculating the PV of two dimensional model output.
"""
import xarray as xr
from . import general as PVG


def remove_y_coordinates(ds):
    ''' removes the y coordinate from a 2D model

    Args:
        ds --> xarray dataset of model output from a 2D model. Can contain
            dimensions of 'Y' or 'Yp1' or both or neither

    Returns:
        ds2d --> copy of ds without dimensions of 'Y' or 'Yp1'

    Notes:
        - If neither 'Y' or 'Yp1' are present the function will just return
            the original dataset.
    '''
    if isinstance(ds, xr.Dataset):
        pass
    elif isinstance(ds, xr.DataArray):
        pass
    else:
        raise TypeError('ds must be of type xr.Dataset or xr.DataArray but is of type {}'.format(type(ds)))

    ds2d = ds
    try:
        ds2d = ds2d.isel({'Y': 0}).drop_vars('Y')
    except ValueError:
        pass

    try:
        ds2d = ds2d.isel({'Yp1': 0}).drop_vars('Yp1')
    except ValueError:
        pass

    return ds2d


def remove_land_points(ds, da_depth=None):
    ''' removes points corresponding to land from a 2D dataset

    Arguments:
        ds --> dataset to remove land points from

    Keyword arguments:
        da_depth --> xr.DataArray containing depth values

    Returns:
        ds --> dataset with land points removed

    Notes:
        - If da_depth is not given the function assumes that the Eastern and
            Western grid points are over land.
    '''

    # If da_depth is provided check whether there are edge points
    if da_depth is not None:
        if not isinstance(da_depth, xr.DataArray):
            raise TypeError('type of da_depth should be xr.DataArray but given object is {}'.format(type(da_depth)))

        if da_depth.isel({'X': 0}) == 0:
            West = True
        else:
            West = False

        if da_depth.isel({'X': -1}) == 0:
            East = True
        else:
            East = False

    else:
        West = True
        East = True

    # Remove the edge points
    if East:
        try:
            xlen = ds.dims['X']
            ds = ds.isel(X=slice(0, xlen - 1))
        except KeyError:
            pass

        try:
            xplen = ds.dims['Xp1']
            ds = ds.isel(Xp1=slice(0, xplen - 1))
        except KeyError:
            pass

    if West:
        try:
            xlen = ds.dims['X']
            ds = ds.isel(X=slice(1, xlen))
        except KeyError:
            pass

        try:
            xplen = ds.dims['Xp1']
            ds = ds.isel(Xp1=slice(1, xplen))
        except KeyError:
            pass

    return ds


def cleanup_dataset(ds, ds_grid):
    ''' applies post-processing to model output

    Args:
        ds --> dataset to be post processed
        ds_grid --> grid dataset corresponding to ds

    Returns:
        None

    Notes:
        - function removes y coordinates and land points, and formats the
            level data of a raw MITgcm netcdf output file.
    '''
    ds2d = remove_y_coordinates(ds)
    ds2d_noedge = remove_land_points(ds2d, da_depth=ds_grid.Depth)
    ds2d_noedge_zformat = PVG.format_vertical_coordinates(ds2d_noedge, ds_grid)

    return ds2d_noedge_zformat


def calc_q(da_vvel, da_rhoanom, da_momVort3, da_f):
    """ Function to calculate PV from constituent inputs.
    """
    raise NotImplementedError()
