#!/usr/bin/env python3
import xarray as xr

def construct_grid_file_name(processor, tile):
    fn = processor + '/grid.' + tile + '.nc'
    return fn


def deconstruct_processor_tile_relation(fn):
    processor = fn.split('/')[0]
    tile = fn.split('.')[1]
    return processor, tile


def is_boundary(depth):
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
    vert = da_vvort * ds_b['dbdz']
    merid = ds_hvort['merid'] * ds_b['dbdy']
    zonal = ds_hvort['zonal'] * ds_b['dbdx']
    q = xr.Dataset()
    q['potVort'] = vert + merid + zonal
    return q
