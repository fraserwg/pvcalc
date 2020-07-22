#!/usr/bin/env python3
"""
########################################################################
slice_script.py
########################################################################

Welcome to slice_script.py, a python utility for calculating PV
from MITgcm generated netcdf files.

Command line options:
-d --> input/output directory
-n --> number of processors to use
-l --> latitude of slice (in m)
-F --> Full Coriolis component
-o --> Name of output file 

########################################################################
"""
import os
from glob import glob
from multiprocessing import Pool
import getopt
import sys
import time
from datetime import datetime
from packaging import version
import xarray as xr
import PVCALC.general as PVG
import PVCALC.slice_ as PVS

if __name__ == '__main__':
    print(__doc__)
    now = datetime.now()
    print('{} {}:{}:{} \n'.format(now.date(), now.hour, now.minute, now.second))


    # Read in command line options
    args = sys.argv[1:]
    opts, args = getopt.getopt(args, 'd:l:n:F:o:')

    for o, a in opts:
        if o == '-d':
            run_folder = a
        elif o == '-l':
            lat = a
        elif o == '-n':
            nprocs = a
        elif o == '-o':
            out_file = a
        elif o == '-F':
            fCoriCos = a
        else:
            raise NotImplementedError('Option {} not supported'.format(o))

    # Force the command lines options into correct format
    try:
        nprocs = int(nprocs)
    except NameError:
        nprocs = os.cpu_count()

    try:
        out_file = os.path.abspath(out_file)
    except NameError:
        out_file = os.path.abspath('./gluPV.nc')

    try:
        lat = float(lat)
    except NameError:
        lat = 400e3

    try:
        fCoriCos = float(fCoriCos)
    except NameError:
        fCoriCos = 0

    try:
        run_folder = os.path.abspath(run_folder)
    except NameError:
        run_folder = os.path.abspath(os.getcwd())

    # Display the options
    print('run folder set to {}'.format(run_folder))
    print('Output will be saved to {}'.format(out_file))
    print('Operating at latitude {:.2f} km'.format(lat*1e-3))
    print('Number of processors used: {} \n'.format(nprocs))

    print(72 * '#' + '\n')


    t0 = time.time()
    # Change directory to the run folder
    os.chdir(run_folder)

    # Each tile has a grid file. Find them and use to determine processor/tile relationships.
    print('Searching for tiles')
    grid_files = glob('mnc*/grid*')
    processor_tile = [PVG.deconstruct_processor_tile_relation(
        fn) for fn in grid_files]

    tiles_in_slice = [elem for elem in processor_tile if PVS.is_tile_in_slice(*elem, lat)]
    ptlat = [[elem, lat, fCoriCos] for elem in tiles_in_slice]

    tsearch = time.time() - t0
    print('Found {} tiles \n'.format(len(ptlat)))

    # Make sure some tiles are found
    if len(ptlat) == 0:
        print('No tiles found, exiting PVCALC.py')
        sys.exit()
    elif len(ptlat) < nprocs:
        nprocs = len(ptlat)
        print('Number of processors reduced to match number of tiles \n')
        print('nprocs = {}'.format(nprocs))


    # Of the tiles found, check which are in the desired slice

    # Calculate the PV in parallel (process based)
    print('Initialising process pool')
    with Pool(nprocs) as p:
        pv_list = p.starmap(PVS.calc_pv_of_tile, ptlat)
    tpool = time.time() - tsearch - t0

    # Merge the processed output
    print('Merging processed output')
    if version.parse(xr.__version__) >= version.parse("0.16"):
        ds_combined = xr.combine_by_coords(pv_list, combine_attrs='override')
    else:
        ds_combined = xr.combine_by_coords(pv_list)
    tcombined = time.time() - tpool - tsearch - t0

    # Save the merged output
    print('Saving merged output \n')
    ds_combined.to_netcdf(out_file)
    tsave = time.time() - tcombined - tpool - tsearch - t0

    print('Processing completed \n')

    print(72 * '#' + '\n')

    # Summarise the outcome
    print('Summary statistics below:')
    print('t_search = {} secs'.format(tsearch))
    print('t_pool   = {} secs'.format(tpool))
    print('t_combined  = {} secs'.format(tcombined))
    print('t_save   = {} secs \n'.format(tsave))

    print(72 * '#' + '\n')
