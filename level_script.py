#!/usr/bin/env python3
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
import general as PVG
import level as PVL
import logging
import logging.handlers
import time

if __name__ == '__main__':
    print(72 * '#')
    print('level_script.py')
    print(72 * '#' + '\n')

    print('Welcome to level_script.py, a python utility for calculating PV')
    print('from MITgcm generated netcdf files.\n')

    print('Command line options:')
    print('-d --> input/output directory')
    print('-n --> number of processors to use')
    print('-l --> model level to operate on')
    print('-F --> Full Coriolis component')
    print('-o --> Name of output file \n')

    print(72 * '#' + '\n')

    # Read in command line options
    args = sys.argv[1:]
    opts, args = getopt.getopt(args, 'd:l:n:F:o:')

    for o, a in opts:
        if o == '-d':
            run_folder = a
        elif o == '-l':
            lvl = a
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
        lvl = int(lvl)
    except NameError:
        lvl = 1

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
    print('Operating on lvl {}'.format(lvl))
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
    ptlvl = [[elem, lvl, fCoriCos] for elem in processor_tile]

    tsearch = time.time() - t0
    print('Found {} tiles \n'.format(len(ptlvl)))

    # Make sure some tiles are found
    if len(ptlvl) == 0:
        print('No tiles found, exiting PVCALC.py')
        sys.exit()

    # Calculate the PV in parallel (process based)
    print('Initialising process pool')
    with Pool(nprocs) as p:
        pv_list = p.starmap(PVL.calc_pv_of_tile, ptlvl)
    tpool = time.time() - tsearch - t0

    # Merge the processed output
    print('Merging processed output')
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
