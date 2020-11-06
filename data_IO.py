'''Functions for reading and writing data from vtk and vtu

Created on Wed April 05 2017

Author: Christopher Strickland
Email: cstric12@utk.edu
'''

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

import os, sys
if (sys.version_info[0] >= 3):
    # Python 3 being used
    PY3 = True
    from pathlib import Path
else:
    # Python 2
    PY3 = False
import numpy as np
import vtk
from vtk.util import numpy_support # Converts vtkarray to/from numpy array
from vtk.numpy_interface import dataset_adapter as dsa


def read_vtk_Structured_Points(filename):
    '''This will read in either Scalar or Vector data!'''

    # Load data
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.Update()
    vtk_data = reader.GetOutput()

    # Get mesh data
    mesh_shape = vtk_data.GetDimensions()
    origin = vtk_data.GetOrigin()
    spacing = vtk_data.GetSpacing()

    # Read in data
    scalar_data = vtk_data.GetPointData().GetScalars()
    if scalar_data is not None:
        np_data = numpy_support.vtk_to_numpy(scalar_data)
        e_data = np.reshape(np_data, mesh_shape[::-1]).squeeze()
        # Indexed [z,y,x] since x changes, then y, then z in the flattened array
        return e_data, origin, spacing
    else:
        vector_data = vtk_data.GetPointData().GetVectors()
        np_data = numpy_support.vtk_to_numpy(vector_data)
        e_data_X = np.reshape(np_data[:,0], mesh_shape[::-1]).squeeze()
        e_data_Y = np.reshape(np_data[:,1], mesh_shape[::-1]).squeeze()
        e_data_Z = np.reshape(np_data[:,2], mesh_shape[::-1]).squeeze()
        # Each of these are indexed via [z,y,x], since x changes, then y, then z
        #   in the flattened array.
        return e_data_X, e_data_Y, e_data_Z, origin, spacing



def read_vtk_Rectilinear_Grid_Vector(filename):
    '''Reads a vtk file with Rectilinear Grid Vector data and TIME info'''

    reader = vtk.vtkRectilinearGridReader()
    reader.SetFileName(filename)
    # Note: if multiple variables are in the same file, you will need to
    #   add the line(s) reader.ReadAllVectorsOn() and/or reader.ReadAllScalarsOn()
    reader.Update()
    vtk_data = reader.GetOutput()
    mesh_shape = vtk_data.GetDimensions() # (xlen, ylen, zlen)
    mesh_bounds = vtk_data.GetBounds() # (xmin, xmax, ymin, ymax, zmin, zmax)

    # Get mesh data
    x_mesh = numpy_support.vtk_to_numpy(vtk_data.GetXCoordinates())
    y_mesh = numpy_support.vtk_to_numpy(vtk_data.GetYCoordinates())
    z_mesh = numpy_support.vtk_to_numpy(vtk_data.GetZCoordinates())

    # Read in vector data
    np_data = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetVectors())
    # np_data is an (n,3) array of n vectors

    # Split and reshape vectors to a matrix
    x_data = np.reshape(np_data[:,0], mesh_shape[::-1]).T
    y_data = np.reshape(np_data[:,1], mesh_shape[::-1]).T
    z_data = np.reshape(np_data[:,2], mesh_shape[::-1]).T
    # Each of these are originally indexed via [z,y,x], since x changes, then y, 
    # then z in the flattened array. Transpose to [x,y,z]

    ##### This code is now hanging indefinitely due to some sort of   #####
    #####   internal vtk dataset_adapter error. Using a workaround... #####

    # In the case of VisIt IBAMR data, TIME is stored as fielddata. Retrieve it.
    #   To do this, wrap the data object and then access FieldData like a dict
    # py_data = dsa.WrapDataObject(vtk_data)
    # if 'TIME' in py_data.FieldData.keys():
    #     time = numpy_support.vtk_to_numpy(py_data.FieldData['TIME'])
    #     assert len(time) == 1, "Currently can only support single time data."
    #     time = time[0]
    # else:
    #     time = None

    # FYI, py_data.PointData is a dict containing the point data - useful if
    #   there are many variables stored in the same file. Also, analysis on this
    #   data (which is a vtk subclass of numpy arrays) can be done by importing
    #   algorithms from vtk.numpy_interface. Since the data structure retains
    #   knowledge of the mesh, gradients etc. can be obtained across irregular points

    ##### Here is the workaround #####

    fielddata = vtk_data.GetFieldData()
    time = None
    for n in range(fielddata.GetNumberOfArrays()):
        # search for 'TIME'
        if fielddata.GetArrayName(n) == 'TIME':
            time = numpy_support.vtk_to_numpy(fielddata.GetArray(n))
            assert len(time) == 1, "Currently can only support single time data."
            time = time[0]
            break
    
    return (x_data, y_data, z_data), (x_mesh, y_mesh, z_mesh), time



def read_vtk_Unstructured_Grid_Points(filename):
    '''This is meant to read mesh data exported from VisIt, where the mesh
    contains only singleton points.'''

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    vtk_data = reader.GetOutput()

    # Cells are specified by a vtk cell type and a list of points that make up
    #   that cell. py_data.CellTypes is a list of numerial cell types.
    #   py_data.Cells is an array that, for each cell in order, lists the number
    #   of points in that cell followed by the index of each cell point. Since
    #   this makes random access to a given cell hard, a third data structure
    #   py_data.CellLocations is specified, which gives the index in py_data.Cells
    #   at which each cell starts.

    # Since we are assuming that the mesh is only singleton points (CellType=1),
    #   we only need the location of the points.
    
    ##### This code is now hanging indefinitely due to some sort of   #####
    #####   internal vtk dataset_adapter error. Using a workaround... #####
    # py_data = dsa.WrapDataObject(vtk_data)
    # points = numpy_support.vtk_to_numpy(py_data.Points) # each row is a 3D point location

    # also get bounds of the mesh (xmin, xmax, ymin, ymax, zmin, zmax)
    # try:
    #     bounds = numpy_support.vtk_to_numpy(py_data.FieldData['avtOriginalBounds'])
    # except AttributeError:
    #     bounds_list = []
    #     for ii in range(points.shape[1]):
    #         bounds_list.append(points[:,ii].min())
    #         bounds_list.append(points[:,ii].max())
    #     bounds = np.array(bounds_list)
    
    ##### Here is the workaround #####

    vtkpoints = vtk_data.GetPoints()
    points = numpy_support.vtk_to_numpy(vtkpoints.GetData())
    # also get bounds of the mesh (xmin, xmax, ymin, ymax, zmin, zmax)
    fielddata = vtk_data.GetFieldData()
    bounds = None
    for n in range(fielddata.GetNumberOfArrays()):
        # search for 'avtOriginalBounds'
        if fielddata.GetArrayName(n) == 'avtOriginalBounds':
            bounds = numpy_support.vtk_to_numpy(fielddata.GetArray(n))
            break
    if bounds is None:
        bounds_list = []
        for ii in range(points.shape[1]):
            bounds_list.append(points[:,ii].min())
            bounds_list.append(points[:,ii].max())
        bounds = np.array(bounds_list)
        
    return points, bounds



def read_2DEulerian_Data_From_vtk(path, simNums, strChoice, xy=False):
    '''This is to read IB2d data, either scalar or vector.'''

    if PY3:
        filename = Path(path) / (strChoice + '.' + str(simNums) + '.vtk')
        data = read_vtk_Structured_Points(str(filename))
    else:
        filename = os.path.normpath(path)+strChoice+'.'+str(simNums)+'.vtk'
        data = read_vtk_Structured_Points(filename)

    if xy:
        # reconstruct mesh
        origin = data[-2]
        spacing = data[-1]
        x = np.arange(data[0].shape[0])*spacing[0]+origin[0]
        y = np.arange(data[0].shape[1])*spacing[1]+origin[1]

    # infer if it was a vector or not and return accordingly
    if len(data) == 3:
        # scalar
        if xy:
            return data[0], x, y
        else:
            return data[0]
    else:
        # vector
        if xy:
            return data[0], data[1], x, y
        else:
            return data[0], data[1]



def read_vtu_mesh_velocity(filename):
    '''This method reads ascii COMSOL velocity data in a vtu or equivalent 
    txt file. It is assumed that the data is on a regular grid.
    
    Returns velocity data and grid as lists.'''

    with open(filename) as f:
        dimension = None
        grid_units = None
        section_flag = None
        grid = []
        vel_data = {}
        for line in f:
            line = line.strip()
            if line[0] == '%':
                # Comment line
                line = line[1:].strip()
                if 'Dimension:' == line[:10]:
                    dimension = int(line[10:].strip())
                    section_flag = None
                elif 'Length unit:' == line[:12]:
                    grid_units = line[12:].strip()
                    print('Grid units are in '+grid_units+'.')
                    section_flag = None
                elif line[:4] == 'Grid':
                    section_flag = 'Grid'
                elif line[:4] == 'Data':
                    section_flag = 'Data'
                elif section_flag == 'Data' and\
                    line[0] in ['x', 'y', 'z', 'u', 'v', 'w']:
                    # point or velocity data
                    section_flag = line[0]
                    section_units = line[2:]
                else:
                    section_flag = None
            else:
                # Non-comment line
                if section_flag == 'Grid':
                    grid.append(np.loadtxt([line]))
                if section_flag in ['x', 'y', 'z']:
                    # Do nothing with point data, already have grid
                    pass
                if section_flag in ['u', 'v', 'w']:
                    # Velocity data. With respect to the grid, this moves in x,
                    #   then y, then z.
                    if section_flag not in vel_data:
                        print(section_flag+' units are '+section_units+'.')
                        vel_data[section_flag] = [np.loadtxt([line])]
                    else:
                        vel_data[section_flag].append(np.loadtxt([line]))
    # Done reading file

    # Gather and parse the data and return
    data = []
    # need to reshape velocity data
    # first, get the length of each dimension of the grid
    dimlen = []
    for dim in grid:
        dimlen.append(len(dim))
    # data is currently a list of 1D arrays. Convert to 2D array, then reshape
    #   into 3D if necessary. x is read across rows of the txt data, so with
    #   row-major ordering this will b3e [z,y,x]. Transpose to [x,y,z].
    for vel in ['u', 'v']:
        # convert to 2D array, replace nan with 0.0
        vel_data[vel] = np.nan_to_num(np.array(vel_data[vel]), copy=False)
    if len(vel_data.items()) == 3:
        # convert to 2D array, replace nan with 0.0
        vel_data['w'] = np.nan_to_num(np.array(vel_data['w']), copy=False)
        # reshape all to [z,y,x] then transpose to [x,y,z]
        for vel in ['u', 'v', 'w']:
            data.append(np.reshape(vel_data[vel], dimlen[::-1]).T)
    else:
        for vel in ['u', 'v']:
            # should already be shaped as [y,x]; transpose to [x,y]
            data.append(vel_data[vel].T)
    # check that the dimensions match up
    for d in data:
        assert d.shape == tuple(dimlen), "Vel dimensions do not match grid!"

    return data, grid
    

                
