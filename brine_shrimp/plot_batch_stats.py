'''
This script is meant to load a list of results and plot the associated
statistics so that distribution shapes can be compared across models.
'''

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dirname = 'data/results/10min_var2.5_N10000'

folder = Path(dirname)

stat_files = [f for f in folder.iterdir() if f.is_file() and f.name[-4:] == '.npz']

data = {}
names = []

for f in stat_files:
    fobj = np.load(f) # returns a dictionary-like file obj. query contents w/ .files
    suffix_idx = f.name.find('_var')
    fname = f.name[:suffix_idx]
    print(fname+' green/blue crossing fractions: {}, {}'.format(
        fobj['g_cross_frac'], fobj['b_cross_frac']
    ))
    names.append(fname)
    data[fname] = fobj

# create dataframe
df = pd.DataFrame(data=[data[fname] for fname in names], index=names, dtype=float)
df.drop(labels=['g_cross_frac', 'b_cross_frac'], axis=1, inplace=True)

# reorder the columns
cols = ['g_mean', 'g_median', 'g_mode', 'g_std', 'g_skew', 'g_kurt',
    'b_mean', 'b_median', 'b_mode', 'b_std', 'b_skew', 'b_kurt']
df = df[cols]

# save it as latex
# requires usepackage{booktabs}
df.to_latex('stat_table.tex')