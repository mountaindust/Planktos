'''
This script is meant to load a list of results and plot the associated
statistics so that distribution shapes can be compared across models.
'''

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

dirname = '10min_var2.5_N10000'

folder = Path(dirname)

stat_files = [f for f in folder.iterdir() if f.is_file() and f.name[-4:] == '.npz']

data = {}

for f in stat_files:
    fobj = np.load(f) # returns a dictionary-like file obj. query contents w/ .files
    suffix_idx = f.name.find('_var')
    fname = f.name[:suffix_idx]
    print(fname+' green/blue crossing fractions: {}, {}'.format(
        fobj['g_cross_frac'], fobj['b_cross_frac']
    ))
    data[fname] = fobj

