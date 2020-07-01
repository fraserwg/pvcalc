import xarray as xr


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
    if type(ds) == xr.Dataset:
        pass
    elif type(ds) == xr.DataArray:
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
