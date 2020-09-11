#!/usr/bin/env python3
""" PVCALC.level

Module contains functions used to calculate the PV of a model level.
"""
from glob import glob
from threading import Thread
from queue import Queue
import xarray as xr
import MITgcmutils.mds as mds
from . import general as PVG


def _create_dataset_list(file_list, variable):
    """ opens the requested datasets and selects the variable of interest

    Arguments:
        file_list (list) -->  List of paths to the files to be opened. Each
            element of the list typically corresponds to the same tile but a
            different time period.
        variable (list or None) --> List of variables to load. If None, will
            load all present.

    Returns:
        ds_list (list) --> List of xarray.Dataset objects.

    """
    if variable is None:
        ds_list = [xr.open_dataset(fn) for fn in file_list]
    elif isinstance(variable, list):
        ds = xr.open_dataset(file_list[0])
        drop_list = [elem for elem in ds.data_vars]
        [drop_list.remove(var) for var in variable]
        ds_list = [xr.open_dataset(fn).drop_vars(drop_list)
                   for fn in file_list]
    else:
        raise TypeError('variable must be either list or None')
    return ds_list

def _select_dataset_levels(ds_list, ds_grid, lvl):
    """ selects the requested dataset level and those surrounding it

    Arguments:
        ds_list (list) -->  List of datasets we are selecting levels for
        ds_grid (xarray.Dataset) --> The grid ds of the tile.
        lvl (int) --> The model level we are selecting.

    Returns:
        ds_list (list) --> List of xarray.Dataset objects.
        ds_grid (xarray.Dataset) --> The grid ds for the tile with the same
            vertical levels of the items in ds_list

    Notes:
        - Variables are selected at the cell centre of the requested level,
            and the cell centres above and below. Variables not on cell centres
            are selected half a cell above and below the requested level.
    """

    Zmd_name = 'Zmd{:06d}'.format(ds_list[0].Nr)
    Zl_name = 'Zld{:06d}'.format(ds_list[0].Nr)
    Zu_name = 'Zud{:06d}'.format(ds_list[0].Nr)

    try:
        [ds.dims[Zmd_name] for ds in ds_list]
        ds_list = [ds.isel({Zmd_name: slice(lvl - 1, lvl + 2)})
                   for ds in ds_list]
        ds_grid = ds_grid.isel({'Z': slice(lvl - 1, lvl + 2)})
    except KeyError:
        pass

    try:
        [ds.dims[Zl_name] for ds in ds_list]
        ds_list = [ds.isel({Zl_name: slice(lvl, lvl + 2)}) for ds in ds_list]
        ds_grid = ds_grid.isel({'Zl': slice(lvl, lvl + 2)})
    except KeyError:
        pass

    try:
        [ds.dims[Zu_name] for ds in ds_list]
        ds_list = [ds.isel({Zu_name: slice(lvl - 1, lvl + 1)})
                   for ds in ds_list]
        ds_grid = ds_grid.isel({'Zu': slice(lvl - 1, lvl + 1)})
    except KeyError:
        pass
    return ds_list, ds_grid


def open_tile(file, tile, processor_dict, lvl=1, variable=None):
    """ Opens all the file files for a particular tile

    Arguments:
        file (str) --> Prefix of file to open, e.g. 'Vorticity'.
        processor (str) --> The processor folder, e.g 'mnc_0001'.
        tile (str) --> The tile file suffix, e.g. 't001'.
        lvl (int) --> The model level to operate on.
        variable (list) --> The variables in the dataset to load, e.g.
            ['UVEL', 'VVEL'] or None.
    """

    processor = processor_dict[tile]
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

    # Open the grid file
    grid_fn = PVG.construct_grid_file_name(processor, tile)
    ds_grid = xr.open_dataset(grid_fn)
    depth = ds_grid['Depth']

    # Open each dataset in the file list, select the vertical levels and rename
    # the vertical levels appropriately
    ds_list = _create_dataset_list(file_list, variable)
    ds_list, ds_grid = _select_dataset_levels(ds_list, ds_grid, lvl)
    ds_list = [PVG.format_vertical_coordinates(ds, ds_grid) for ds in ds_list]

    # Check if we have to join along the 'T' axis and do so if required
    if len(file_list) == 1:
        ds_var = ds_list[0]
    elif len(file_list) >= 1:
        ds_var = xr.concat(ds_list, dim='T')

    # Establish any land boundary points and remove them.
    ds_var = remove_boundary_points(ds_var, depth)

    return ds_var


def remove_boundary_points(ds, depth):
    """ Determines whether a tile has a solid boundary and removes those points
    if so

    Arguments:
        ds --> xarray dataset containing variable of interest
        depth --> xarray dataarray containing the depth corersponding to this
            tile

    Returns:
        ds --> original dataset with any land points removed.
    """
    # Determine boundary points
    North, South, East, West = PVG.is_boundary(depth)

    # Remove boundary points if necessary
    if North:
        try:
            ylen = ds.dims['Y']
            ds = ds.isel(Y=slice(0, ylen - 1))
        except KeyError:
            pass

        try:
            yplen = ds.dims['Yp1']
            ds = ds.isel(Yp1=slice(0, yplen - 1))
        except KeyError:
            pass

    if South:
        try:
            ylen = ds.dims['Y']
            ds = ds.isel(Y=slice(1, ylen))
        except KeyError:
            pass

        try:
            yplen = ds.dims['Yp1']
            ds = ds.isel(Yp1=slice(1, yplen))
        except KeyError:
            pass

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


def grad_b(ds_rho, rho_ref):
    """ Calculates the gradient of the buoyancy field from the density field.

    Arguments:
        ds_rho --> xarray dataset containing 'RHOAnoma', the density anomaly.
            Should be opened using open_tile
        rho_ref --> The reference density which added to the dat in RHOAnoma
            gives the total density. Should be opened using mds then sliced.

    Returns:
        ds_b --> dataset containing the buoyancy gradients and buoyancy.

    Notes:
        rho_ref is NOT the density of the reference level. That is set to
            1000 kg/m^3 in the below function
    """
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
    """ Calculates the horizontal vorticity.

    Arguments:
        ds_vel --> xarray dataset of velocities as opened by open_tile. Should
            contain the ketys 'UVEL', 'VVEL' and 'WVEL' defined on the C-grid.
        fCoriCos --> Non-traditional component of the Coriolis force. Either a
            float or an xarray dataarray or numpy array defined on the 'X' and
            'Y' points of the tile.

    Returns:
        ds_vort --> xarray dataset containing the horizontal components of the
            absolute vorticity.
    """
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
    """ Calculates the vertical component of the absolute vorticity.

    Arguments:
        ds_vert --> xarray dataset containing the vertical component of the
            relative vorticity. Contains the key 'momVort3'.
        ds_grid --> xarray dataset containing the grid data of the domain. Must
            contain the key 'fCoriG' which is the vertical component of the
            planetary vorticity defined at the cell corners.

    Returns:
        da_vort --> xarray dataarray containing the vertical component of the
            absolute vorticity.
    """
    ds_vert = ds_vert.isel({'Z': 1})
    da_vort = xr.DataArray()
    da_vort = ds_vert['momVort3'] + ds_grid['fCoriG']
    da_vort = da_vort.interp({'Xp1': ds_grid['X'], 'Yp1': ds_grid['Y']})
    da_vort = da_vort.drop_vars(['Xp1', 'Yp1'])
    return da_vort


def calc_pv_of_tile(tile, processor_dict, lvl, fCoriCos):
    """ Function to calculate the PV of the tile and level.
    Arguments:
        proc_tile (tuple) --> tuple of processor and tile strings, e.g.
            ('mnc_0001', 't003')
        lvl (int) --> model level to calculate PV of.
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
    pt = (processor_dict[tile], tile)
    ds_rho = open_tile('Rho', tile, processor_dict, lvl=lvl)
    rho_ref = mds.rdmds('RhoRef')[slice(lvl - 1, lvl + 2)]
    ds_vert = open_tile('Vorticity', tile, processor_dict, lvl=lvl)
    ds_grid = open_tile('grid', tile, processor_dict, lvl=lvl)
    ds_vel = open_tile('Velocity', tile, processor_dict, lvl=lvl)

    # Calculate vorticity and buoyancy gradients in parallel (thread based).
    que = Queue()

    b_thread = Thread(target=lambda q, ds_rho, rho_ref: q.put(
        grad_b(ds_rho, rho_ref)), args=(que, ds_rho, rho_ref))
    vv_thread = Thread(target=lambda q, ds_vert, ds_grid: q.put(
        abs_vort(ds_vert, ds_grid)), args=(que, ds_vert, ds_grid))
    hv_thread = Thread(target=lambda q, ds_vel, fCoriCos: q.put(
        hor_vort(ds_vel, fCoriCos)), args=(que, ds_vel, fCoriCos))

    thread_list = [b_thread, vv_thread, hv_thread]
    [thread.start() for thread in thread_list]
    [thread.join() for thread in thread_list]

    # Extract variable from the que
    da_vvort, ds_b, ds_hvort = _drain_the_component_que(que)

    # Calculate q
    q = PVG.calc_q(ds_b, da_vvort, ds_hvort)
    q.attrs = ds_vel.attrs

    # Save to a netcdf file
    proc, tile = pt
    outname = './' + proc + '/PV.' + tile + '.nc'
    q.to_netcdf(outname)
    return q


def _drain_the_component_que(que):
    """ Extracts the bouyancy and vorticity xarray objects from que.

    Arguments:
        que --> queue.Queue object  containing jobs calculating the horizontal
            and vertical components of vorticity and the buoyancy

    Returns:
        da_vvort --> xarray dataarray containing the vertical absolute
            vorticity
        ds_b --> xarray dataset containing buoyancy and its gradients
        ds_hvort --> xarray dataset containing the horizontal vorticity
            components

    Notes:
        This function is very hacky and specialised. It shouldn't normally be
        accessed by the user.
    """
    while not que.empty():
        result = que.get()
        if isinstance(result, xr.DataArray):
            da_vvort = result
        else:
            try:
                result['dbdz']
                ds_b = result
            except KeyError:
                ds_hvort = result
    return da_vvort, ds_b, ds_hvort
