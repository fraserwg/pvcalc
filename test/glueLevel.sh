#!/bin/bash
#SBATCH --job-name=PVCALC
#SBATCH --output=glue.out
#SBATCH --time=15:00
#SBATCH --ntasks=16
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=fraser.goldsworth@physics.ox.ac.uk
#SBATCH -p priority-ocean

export DIR=./test_data/

level_script.py -d $DIR -n 16 -l 2 -o $DIR/gluPVlvl.nc -F 0.00015 > glueLvl.out
