'''
Run to remove output from Jupyter notebook before committing
'''

import os

os.system('jupyter nbconvert --clear-output proj_and_sld_mv_test.ipynb --inplace')