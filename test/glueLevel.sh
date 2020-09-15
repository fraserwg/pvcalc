#!/bin/bash
#SBATCH --job-name=PVCALC
#SBATCH --output=glue.out
#SBATCH --time=15:00
#SBATCH --ntasks=16
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=fraser.goldsworth@physics.ox.ac.uk
#SBATCH -p priority-ocean

export DIR=./test_data/

level_script.py --dir $DIR -n 4 --lvl 2 --out $DIR/gluPVlvl.nc --fCoriCos 0.00015 > glueLvl.out
