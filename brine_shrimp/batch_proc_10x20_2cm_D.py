'''
Batch script for running multiple simulations across diffusivity
'''

from pathlib import Path
import os

def main():
    datafile = './data/comsol/Velocity_10x20_2cm.vtu'
    D_list = [0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25]
    for D in D_list:
        prefix = '10x20_2cm_'
        suffix = '_N10000'
        name = prefix + 'D{}'.format(D) + suffix
        cmd = 'python comsol_shrimp.py -N 10000 -t 600 -o {} -d {} -D {}'.format(name, datafile, D)
        print('Now processing {}.'.format(name))
        os.system(cmd)

if __name__ == "__main__":
    main()
