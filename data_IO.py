'''Functions for reading and writing data

Created on Wed April 05 2017

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
'''

__author__ = "Christopher Strickland"
__email__ = "wcstrick@live.unc.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

import os
from pathlib import Path
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
    x_data = np.reshape(np_data[:,0], mesh_shape[::-1])
    y_data = np.reshape(np_data[:,1], mesh_shape[::-1])
    z_data = np.reshape(np_data[:,2], mesh_shape[::-1])
    # Each of these are indexed via [z,y,x], since x changes, then y, then z
    #   in the flattened array.

    # In the case of VisIt IBAMR data, TIME is stored as fielddata. Retrieve it.
    #   To do this, wrap the data object and then access FieldData like a dict
    py_data = dsa.WrapDataObject(vtk_data)
    if 'TIME' in py_data.FieldData.keys():
        time = numpy_support.vtk_to_numpy(py_data.FieldData['TIME'])
        assert len(time) == 1, "Currently can only support single time data."
        time = time[0]
    else:
        time = None
    # FYI, py_data.PointData is a dict containing the point data - useful if
    #   there are many variables stored in the same file. Also, analysis on this
    #   data (which is a vtk subclass of numpy arrays) can be done by importing
    #   algorithms from vtk.numpy_interface. Since the data structure retains
    #   knowledge of the mesh, gradients etc. can be obtained across irregular points

    return (x_data, y_data, z_data), (x_mesh, y_mesh, z_mesh), time



def read_vtk_Unstructured_Grid_Points(filename):
    '''This is meant to read mesh data exported from VisIt, where the mesh
    contains only singleton points.'''

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    vtk_data = reader.GetOutput()
    
    py_data = dsa.WrapDataObject(vtk_data)
    points = numpy_support.vtk_to_numpy(py_data.Points) # each row is a 3D point location
    
    # Cells are specified by a vtk cell type and a list of points that make up
    #   that cell. py_data.CellTypes is a list of numerial cell types.
    #   py_data.Cells is an array that, for each cell in order, lists the number
    #   of points in that cell followed by the index of each cell point. Since
    #   this makes random access to a given cell hard, a third data structure
    #   py_data.CellLocations is specified, which gives the index in py_data.Cells
    #   at which each cell starts.

    # Since we are assuming that the mesh is only singleton points (CellType=1),
    #   we only need the location of the points.

    # also get bounds of the mesh (xmin, xmax, ymin, ymax, zmin, zmax)
    try:
        bounds = numpy_support.vtk_to_numpy(py_data.FieldData['avtOriginalBounds'])
    except AttributeError:
        bounds_list = []
        for ii in range(points.shape[1]):
            bounds_list.append(points[:,ii].min())
            bounds_list.append(points[:,ii].max())
        bounds = np.array(bounds_list)

    return points, bounds



def read_2DEulerian_Data_From_vtk(path, simNums, strChoice, xy=False):
    '''This is to read IB2d data, either scalar or vector.'''

    filename = Path(path) / (strChoice + '.' + str(simNums) + '.vtk')
    data = read_vtk_Structured_Points(str(filename))

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
