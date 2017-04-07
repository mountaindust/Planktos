'''Functions for reading and writing data

Created on Wed April 05 2017

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
'''

__author__ = "Christopher Strickland"
__email__ = "wcstrick@live.unc.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

import os
import numpy as np

def read_Eulerian_Data_From_vtk(path,simNums,strChoice,first):

    analysis_path = os.getcwd()  # Stores current directory path        
        
    os.chdir(path)               # cd's into viz_IB2d folder
    
    filename = strChoice + '.' + str(simNums) + '.vtk'

    # Stores grid resolution from .vtk file
    Nx = np.genfromtxt(filename, skip_header=5, usecols=(1),max_rows=1)
    
    # Stores desired Eulerian data
    e_data = np.genfromtxt(filename, skip_header=14, 
                            usecols=range(0,int(Nx)), max_rows=int(Nx))

    if first==1:
        
        # Stores Eulerian grid spacing
        dx =  np.genfromtxt(filename, skip_header=8, usecols=(1), 
                            max_rows=1)
    
        # Stores grid values (NOTE: yGrid = xGrid_Transpose)
        x = np.zeros(int(Nx))
        for i in range(1, int(Nx)):        
            x[i] = x[i-1]+dx

        os.chdir(analysis_path)     # Path to working directory

        return e_data,x,x
    
    else:

        os.chdir(analysis_path)     # Path to working directory

        return e_data

def read_Eulerian_Velocity_Field_vtk(path, simNums):

    analysis_path = os.getcwd()  # Stores current directory path        
    
    os.chdir(path)               # cd's into viz_IB2d folder
    
    filename = 'u.' + str(simNums) + '.vtk'

    # Stores grid resolution from .vtk file
    Nx = np.genfromtxt(filename, skip_header=5, usecols=(1),max_rows=1)
    
    # Stores desired Eulerian data
    e_data_X = np.genfromtxt(filename, skip_header=13,
                                usecols=range(0,3*int(Nx),3), max_rows=int(Nx))
    e_data_Y = np.genfromtxt(filename, skip_header=13,
                                usecols=range(1,3*int(Nx),3), max_rows=int(Nx))

    os.chdir(analysis_path)     # Path to working directory

    return e_data_X, e_data_Y