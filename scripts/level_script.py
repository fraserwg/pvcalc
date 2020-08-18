#!/usr/bin/env python3
"""
##############################################################################
level_script.py
##############################################################################

Welcome to level_script.py, a python utility for calculating PV from
MITgcm generated netCDF files.

##############################################################################
"""

import os
import argparse
from glob import glob
from multiprocessing import Pool
import getopt
import sys
import time
from datetime import datetime
from packaging import version
import xarray as xr
import PVCALC.general as PVG
import PVCALC.level as PVL

if __name__ == '__main__':
    print(__doc__)

    parser = argparse.ArgumentParser(prog='level_script.py')

    parser.add_argument('-n', help='Number of processes to use',
                        type=int, default=1, dest='nprocs')
    parser.add_argument('--dir', help='Working directory',
                        type=os.path.abspath, default='./', dest='run_folder')
    parser.add_argument('--out', help='Name of output file',
                        type=os.path.abspath, default='./gluPV.nc', dest='out_file')
    parser.add_argument('--lvl', help='Vertical level to operate on',
                        type=int, default=1, dest='lvl')
    parser.add_argument('--fCoriCos', help='Component of the complete Coriolis Force',
                        type=float, default=0, dest='fCoriCos')

    cl_args = parser.parse_args()
    parser.print_help()
    locals().update(vars(cl_args))
    if nprocs == -1:
        nprocs = os.cpu_count()

    print()
    print(78 * '#' + '\n')

    # Display the options
    now = datetime.now()
    print('{} {}:{}:{} \n'.format(now.date(), now.hour, now.minute, now.second))
    print('run folder set to {}'.format(run_folder))
    print('Output will be saved to {}'.format(out_file))
    print('Operating on lvl {}'.format(lvl))
    print('Number of processors used: {} \n'.format(nprocs))

    print(78 * '#' + '\n')


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
