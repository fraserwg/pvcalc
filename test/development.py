#%%
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
from MITgcmutils import mds

from importlib import reload
# %%
nprocs = 1
run_folder = os.path.abspath('./test_data')
mnc_dir = 'mnc_*'
out_file = os.path.abspath('./gluPV.nc')
lvl = 9
fCoriCos = 1e-5

# %%
os.chdir(run_folder)

print('Searching for tiles')
grid_glob = mnc_dir + '/grid*'
grid_files = glob(grid_glob)
processor_tile = [PVG.deconstruct_processor_tile_relation(fn) for fn in grid_files]
processor_dict = {tile: processor for processor, tile in processor_tile}

calc_pv_args = [(tile, processor_dict, lvl, fCoriCos) for _, tile in processor_tile]
ptlvl = [[elem, lvl, fCoriCos] for elem in processor_tile]

print('Found {} tiles \n'.format(len(ptlvl)))

# %%
reload(PVL)

with Pool(nprocs) as p:
    pv_list = p.starmap(PVL.calc_pv_of_tile, calc_pv_args)

#%%
pt = ('mnc_0001', 't005')
ds_grid = PVL.open_tile('grid', *pt, lvl=lvl)

# %%
ds_rho = PVL.open_tile('Rho', 't001', processor_dict, lvl)
ds_rho
rho_ref = PVL.open_rho_ref(lvl, ds_rho)
# %%

PVL.grad_b(ds_rho, rho_ref, processor_dict, lvl)
# %%
