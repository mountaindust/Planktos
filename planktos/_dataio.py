'''Functions for reading and writing data from vtk, vtu, vertex files, and stl.
These are low-level functions that are probably unnecessary to use directly,
unless you just want to directly get numpy arrays from datasets.

If you are trying to load data into an Environment class, use the relevant
loader methods in Environment.

Created: Wed April 05 2017

Author: Christopher Strickland

Email: cstric12@utk.edu
'''

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

import os
import json
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import vtk
from vtk.util import numpy_support # Converts vtkarray to/from numpy array
# from vtk.numpy_interface import dataset_adapter as dsa
import pyvista as pv

try:
    from stl import mesh as stlmesh
    STL = True
except ModuleNotFoundError:
    STL = False

try:
    import netCDF4 as nc
    NETCDF = True
except ModuleNotFoundError:
    NETCDF = False


#############################################################################
#                                                                           #
#                          DEPENDENCY DECORATORS                            #
#                                                                           #
#############################################################################


def stl_dep(func):
    '''Decorator for STL readers to check package import.'''
    def wrapper(*args, **kwargs):
        __doc__ = func.__doc__
        if not STL:
            print("Cannot read STL file: numpy-stl library not installed.")
            raise RuntimeError("Cannot read STL file: numpy-stl library not found in data_IO. "+
                               "Please install numpy-stl with conda -c conda-forge or using pip.")
        else:
            return func(*args, **kwargs)
    wrapper.__doc__ = func.__doc__
    return wrapper

def netcdf_dep(func):
    '''Decorator for NetCDF readers to check package import.'''
    def wrapper(*args, **kwargs):
        __doc__ = func.__doc__
        if not NETCDF:
            print("Cannot read NetCDF file: NetCDF4 library not installed.")
            raise RuntimeError("Cannot read NetCDF file: NetCDF4 library not found in data_IO. "+
                               "Please install NetCDF4 with conda or using pip.")
        else:
            return func(*args, **kwargs)
    wrapper.__doc__ = func.__doc__
    return wrapper


#############################################################################
#                                                                           #
#                               DATA READERS                                #
#                                                                           #
#############################################################################


def read_vtk_Structured_Points(filename):
    '''Read in either Scalar or Vector data from an ascii VTK Structured Points 
    file using the VTK Python library.

    Used by read_2DEulerian_Data_From_vtk.

    Note: if the file has both scalar and vector data, only the scalar data
    will be returned.
    
    Parameters
    ----------
    filename : string
        path and filename of the VTK file

    Returns
    -------
    e_data : array, indexed [z,y,x] (scalar data only)
    e_data_X, e_data_Y, e_data_Z : arrays, indexed [z,y,x] (vector data only)
        one array for each component of the vector field: X, Y, and Z respectively
    origin : tuple
        origin field of VTK
    spacing : tuple
        spacing of grid
    '''

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
    '''Reads an ascii VTK file with Rectilinear Grid Vector data and TIME info
    using the VTK Python library.

    Parameters
    ----------
    filename : string
        path and filename of the VTK file

    Returns
    -------
    list of arrays
        vector data as numpy arrays, one array for each dimension of the vector
        in order of x, y, z. Each array is indexed as [x,y,z]
    list of arrays
        1D arrays of grid points in the x, y, and z directions
    time : float
    '''

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
    
    return [x_data, y_data, z_data], [x_mesh, y_mesh, z_mesh], time



def read_vtk_time_only(filename, nbytes=4096):
    '''Read just the TIME field-data value from a legacy VTK file's header.

    Legacy VTK writes ``FIELD FieldData`` immediately after the ``DATASET``
    line, so a TIME entry lands within the first few hundred bytes -- ahead of
    the coordinate arrays and long before POINT_DATA. Recovering it therefore
    costs one small header read no matter how large the file is, which is what
    makes it practical to timestamp an entire dump series up front: dynamic
    loading needs the whole timeline before it can slice windows out of it, but
    it must not have to parse every dump to get it.

    The header of a legacy VTK file is ASCII even when the data that follows is
    BINARY, so this works for both.

    Parameters
    ----------
    filename : string
        path and filename of the VTK file
    nbytes : int, default=4096
        how many bytes of the header to scan

    Returns
    -------
    float, or None if no single-valued TIME entry was found in the scanned
        region. None is not proof that the file is untimed -- the format also
        permits FIELD data inside the POINT_DATA/CELL_DATA sections, past the
        scanned region -- so a caller that needs certainty should fall back to
        read_vtk_Rectilinear_Grid_Vector for that file.
    '''

    with open(filename, 'rb') as f:
        head = f.read(nbytes).decode('utf-8', errors='ignore')
    lines = head.splitlines()

    for n, line in enumerate(lines):
        # A FIELD array is declared as "<name> <numComponents> <numTuples>
        # <dataType>" with the values on the following line(s). Require the
        # single-valued form, matching the assertion in the full reader.
        parts = line.split()
        if len(parts) == 4 and parts[0] == 'TIME' and parts[1] == '1' \
                and parts[2] == '1':
            # Only trust the value line if it is not the final line in the
            # buffer, which may have been cut mid-number by the read limit.
            if n + 1 >= len(lines) - 1:
                return None
            try:
                return float(lines[n+1].split()[0])
            except (ValueError, IndexError):
                return None

    return None



def read_vtk_Unstructured_Grid_Points(filename):
    '''Read immersed mesh data from an ascii Unstructured Grid VTK file, such as
    those exported from VisIt. Uses the VTK Python library. The mesh should 
    contain only singleton points (vertices).
    
    Parameters
    ----------
    filename : string
        path and filename of the VTK file

    Returns
    -------
    points : array
        each row is a vertex
    bounds : array
        bounds field data
    '''

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    vtk_data = reader.GetOutput()

    # Cells are specified by a vtk cell type and a list of points that make up
    #   that cell. py_data.CellTypes is a list of numerical cell types.
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



def read_vtu_Unstructured_Grid_Points_FEM(filename):
    '''Read fluid velocity data from an xml Unstructured Grid VTU file, as 
    results from a comsol finite element mesh. Uses the VTK Python library.
    
    Parameters
    ----------
    filename : string
        path and filename of the VTK file

    Returns
    -------
    points : Nx3 array
        each row is the 3D coords of a vertex in the FEM mesh
    data : list of arrays
        each array is velocity data at all mesh points for a given time
    data_names : list of strings
        names of each data array in data
    '''

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    vtk_data = reader.GetOutput()

    # Get location of FEM meshpoints
    vtkpoints = vtk_data.GetPoints()
    points = numpy_support.vtk_to_numpy(vtkpoints.GetData()) #Nx3 array
    # Get point data
    vtkpoint_data = vtk_data.GetPointData()
    # Get number of DataArrays corresponding to point data
    data_size = vtkpoint_data.GetNumberOfArrays()
    # In COMSOL, each three arrays correspond to the x, y, z components of velocity
    #   at each mesh point, with each array being length N where N is the
    #   number of mesh points. Each set of three is a different time point.
    #   So total number of time points is data_size/3.
    data = []; data_names = []
    for n in range(data_size):
        data_array = vtkpoint_data.GetArray(n)
        data.append(numpy_support.vtk_to_numpy(data_array))
        data_names.append(vtkpoint_data.GetArrayName(n))
        
    return points, data, data_names



def read_vtm_series(filename):
    '''Read a ParaView ``.vtm.series`` index, mapping member files to times.

    A ``.vtm.series`` is plain JSON and holds no field data at all -- it is the
    declaration of a time series, listing one member file (typically a ``.vtm``
    manifest) per timestep together with the simulation time it represents::

        {"file-series-version" : "1.0",
         "files" : [{"name" : "case_787.vtm", "time" : 7.5}, ...]}

    Reading it costs a single small parse and needs no VTK, which is what makes
    it usable as the timeline source for dynamic loading: windows can only be
    sliced out of a series whose timestamps are all known up front, and paying a
    full parse per dump to discover them defeats the purpose.

    Nothing is checked against the filesystem here. The listed files may not all
    exist -- a truncated or interrupted export is ordinary -- and deciding what
    to do about that is the caller's policy, not this function's.

    Parameters
    ----------
    filename : string or Path
        path and filename of the .vtm.series file

    Returns
    -------
    files : list of Path
        member files in the order the series declares them, resolved relative to
        the directory holding the series file. The order is preserved rather than
        sorted by time; a series that declares times out of order is reported as
        written.
    times : ndarray of float
        time for each entry of files. An entry the series gave no usable time for
        is NaN rather than an error, so that a caller can fall back to a per-file
        time source (e.g. read_vtm_manifest) for just those entries.
    '''

    filename = Path(filename)
    with open(filename, 'r') as f:
        series = json.load(f)

    if 'files' not in series:
        raise ValueError("No 'files' entry in series index {}.".format(filename))

    files = []; times = []
    for entry in series['files']:
        if 'name' not in entry:
            warnings.warn("Skipping an entry with no 'name' in series index "+
                          "{}.".format(filename), UserWarning)
            continue
        files.append(filename.parent / entry['name'])
        try:
            times.append(float(entry['time']))
        except (KeyError, TypeError, ValueError):
            times.append(np.nan)

    return files, np.array(times, dtype=float)



def read_vtm_manifest(filename):
    '''Read a VTK XML MultiBlockDataSet (``.vtm``) manifest.

    A ``.vtm`` contains no data of its own. It is a small XML file naming the
    child files that together make up one dataset -- for a finite-volume export,
    typically the interior volume plus one file per boundary patch::

        <vtkMultiBlockDataSet>
          <DataSet name='internal' file='case_787/internal.vtu' />
          <Block name='boundary'>
            <DataSet name='inlet' file='case_787/boundary/inlet.vtp' />
            ...

    It is parsed here with the standard library rather than
    ``vtkXMLMultiBlockDataReader``, which would read every child file -- tens of
    megabytes -- merely to report their names. Resolving names is exactly what is
    wanted cheaply: it lets a caller settle at construction which timesteps are
    actually present on disk, instead of discovering a missing file at the window
    slide that needs it, arbitrarily deep into a streaming run.

    Child paths are resolved but not checked for existence; see read_vtm_series.

    Parameters
    ----------
    filename : string or Path
        path and filename of the .vtm file

    Returns
    -------
    datasets : dict of Path, keyed by dataset name
        every ``DataSet`` in the manifest, flattened across any ``Block``
        nesting, resolved relative to the directory holding the manifest. Names
        are assumed unique within a manifest (they are for writers such as
        OpenFOAM's foamToVTK); a repeat keeps the first and warns.
    time : float or None
        the ``TimeValue`` field data entry, if the manifest carries one. This is
        a per-file fallback for a series index that is missing or incomplete.
    '''

    filename = Path(filename)
    root = ET.parse(filename).getroot()

    datasets = {}
    for elem in root.iter('DataSet'):
        name = elem.get('name')
        child = elem.get('file')
        if name is None or child is None:
            # A DataSet may legally carry its data inline instead of naming a
            # file; there is nothing for us to resolve in that case.
            warnings.warn("Skipping a DataSet with no name/file attribute in "+
                          "manifest {}.".format(filename), UserWarning)
            continue
        if name in datasets:
            warnings.warn("Duplicate dataset name '{}' in manifest {}; ".format(
                          name, filename)+"keeping the first.", UserWarning)
            continue
        datasets[name] = filename.parent / child

    time = None
    for elem in root.iter('DataArray'):
        if elem.get('Name') == 'TimeValue' and elem.text is not None:
            try:
                time = float(elem.text.split()[0])
            except (ValueError, IndexError):
                time = None
            break

    return datasets, time



def read_vtkxml_cell_data(filename, arrays=('U',), cell_centers=True):
    '''Read cell data, and the cell-center lattice it lives on, from a VTK XML
    unstructured grid (``.vtu``) or polydata (``.vtp``) file.

    Finite-volume solvers store fields per cell rather than per point, so the
    point data of such a file is typically empty and ``GetPoints()`` returns cell
    *corners* -- the wrong lattice to interpolate a cell field on. This reads
    ``GetCellData()`` and, by default, returns the cell centers to go with it.

    Only the requested arrays are read. That matters for streaming: an
    unstructured-grid file repeats its full mesh on every timestep even when the
    mesh never moves, so most of what is on disk is geometry that is already
    known, and the field arrays that are not wanted are pure waste on top of it.
    Deselecting them is the cheap part of that saving; the geometry itself cannot
    be skipped through this reader.

    Parameters
    ----------
    filename : string or Path
        path and filename of the .vtu or .vtp file
    arrays : iterable of string, or None, default=('U',)
        names of the cell data arrays to read. Any other array present in the
        file is deselected before reading. A requested name that the file does
        not carry warns and is absent from the result. If None, read everything.
    cell_centers : bool, default=True
        whether to compute and return the cell centers. Skip it when the mesh is
        known to be static and the lattice has already been established from an
        earlier file in the series.

    Returns
    -------
    centers : Nx3 ndarray, or None if cell_centers is False
        center of each cell, in the file's own cell order -- which is not
        necessarily lexicographic in the coordinates, even for a grid-like mesh
    data : dict of ndarray, keyed by array name
        one entry per array read, in the same cell order as centers. Shape is
        (N,) for a scalar field and (N,3) for a vector field.
    time : float or None
        the ``TimeValue`` field data entry, if the file carries one
    '''

    filename = Path(filename)
    if not filename.is_file():
        # vtk's XML readers report a missing file only on stderr and then hand
        # back an empty dataset, which would surface much later as a confusing
        # shape mismatch.
        raise FileNotFoundError("File {} not found!".format(str(filename)))

    if filename.suffix == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif filename.suffix == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    else:
        raise ValueError("Unsupported extension '{}': ".format(filename.suffix)+
                         "expected .vtu (unstructured grid) or .vtp (polydata).")
    reader.SetFileName(str(filename))

    # Array selection has to happen against the file's metadata, before the bulk
    # read is triggered by Update().
    reader.UpdateInformation()
    if arrays is not None:
        arrays = tuple(arrays)
        selection = reader.GetCellDataArraySelection()
        available = [selection.GetArrayName(n) for n in
                     range(selection.GetNumberOfArrays())]
        for name in available:
            if name not in arrays:
                selection.DisableArray(name)
        missing = [name for name in arrays if name not in available]
        if len(missing) > 0:
            warnings.warn("Cell data array(s) {} not found in {}. ".format(
                          missing, str(filename))+
                          "Present: {}.".format(available), UserWarning)
    reader.Update()
    vtk_data = reader.GetOutput()

    # Read the cell data
    vtkcell_data = vtk_data.GetCellData()
    data = {}
    for n in range(vtkcell_data.GetNumberOfArrays()):
        name = vtkcell_data.GetArrayName(n)
        data[name] = numpy_support.vtk_to_numpy(vtkcell_data.GetArray(n))

    # Get the cell centers, which are the points this data is specified at
    if cell_centers:
        center_filter = vtk.vtkCellCenters()
        center_filter.SetInputData(vtk_data)
        center_filter.Update()
        centers = numpy_support.vtk_to_numpy(
            center_filter.GetOutput().GetPoints().GetData())
    else:
        centers = None

    # Time, if the writer recorded one. Same workaround as elsewhere in this
    # module: walk the field data by name rather than wrapping the dataset.
    fielddata = vtk_data.GetFieldData()
    time = None
    for n in range(fielddata.GetNumberOfArrays()):
        if fielddata.GetArrayName(n) == 'TimeValue':
            time = numpy_support.vtk_to_numpy(fielddata.GetArray(n))
            assert len(time) == 1, "Currently can only support single time data."
            time = float(time[0])
            break

    return centers, data, time



def read_2DEulerian_Data_From_vtk(path, simNum, strChoice, xy=False):
    '''Reads ascii Structured Points VTK files using the Python VTK library,
    where the file contains 2D IB2d data, either scalar or vector. 
    
    The call signature is set up for easy looping over the sort of file name 
    structure used by IB2d.
    
    Parameters
    ----------
    path : string
        directory containing the vtk files
    simNum : string
        sim number as a string, as given in the filename (with leading zeros)
    strChoice : string
        prefix on the filenames. typically 'u' or 'uX' or 'uY' in IB2d
    xy : bool, default=False
        if True, also return mesh data

    Returns
    -------
    data : one or two arrays
        one array if scalar data, two arrays (x,y) if vector data. NOTE: the
        data is indexed [y,x] so will need to be transposed.
    x, y : mesh data as 1D arrays, only if xy=True
        contains the spatial mesh points in the x and y directions respectively
    '''

    filename = Path(path) / (strChoice + '.' + str(simNum) + '.vtk')
    data = read_vtk_Structured_Points(str(filename))

    if xy:
        # reconstruct mesh
        origin = data[-2]
        spacing = data[-1]
        # data[0,1] are indexed [y,x]!
        x = np.arange(data[0].shape[1])*spacing[0]+origin[0]
        y = np.arange(data[0].shape[0])*spacing[1]+origin[1]

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
    '''Reads ascii COMSOL velocity data in a vtu or equivalent text file. It is 
    assumed that the data is on a regular grid. Currently, there is no support 
    for multiple time points, so the file must contain data from only a single 
    time.

    This is a work-around for the weird vtu ascii file that COMSOL is spitting 
    out. VTU is supposed to be a VTK XML file extension, which should be 
    readable using a vtk.vtkXMLUnstructuredGridReader object, but it dies 
    immediately because the vtu I've got isn't xml at all. We need to fix this.

    Parameters
    ----------
    filename : string
        path and filename of the VTU file
    
    Returns
    -------
    data : list of arrays
        one array for each spatial component of velocity, where each element of
        an array is a gridpoint. arrays are indexed [x,y,z]
    grid : list of arrays
        spatial grid in each direction (1D arrays)
    '''

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
        try:
            for vel in ['u', 'v', 'w']:
                data.append(np.reshape(vel_data[vel], dimlen[::-1]).T)
        except ValueError as e:
            print("If this is a cannot reshape array error, it is likely because "+
            "multiple time points are included in the vtu. This is not supported!")
            raise e
    else:
        for vel in ['u', 'v']:
            # should already be shaped as [y,x]; transpose to [x,y]
            data.append(vel_data[vel].T)
    # check that the dimensions match up
    for d in data:
        assert d.shape == tuple(dimlen), "Vel dimensions do not match grid!"

    return data, grid
    


@stl_dep
def read_stl_mesh(filename, unit_conv=None):
    '''Import a mesh from an stl file and return the vertex information as
    an Nx3x3 array along with the maximum vector length. Uses the numpy-stl 
    library.
    
    Parameters
    ----------
    filename : string
        path and filename of the STL file
    unit_conv : float, optional
            scalar to multiply the mesh by in order to convert units

    Returns
    -------
    array of shape Nx3x3
        Each row is a mesh element consisting of three 3D points. For a given
        row n, this is represented as a 3x3 matrix where each row is a point 
        and each column is a spatial dimension.
    float
        Length of the longest side of any triangle
    '''
    mesh = stlmesh.Mesh.from_file(filename)
    # find maximum segment length
    max_len = np.concatenate((np.linalg.norm(mesh.v1 - mesh.v0, axis=1),
                             np.linalg.norm(mesh.v2 - mesh.v1, axis=1),
                             np.linalg.norm(mesh.v0 - mesh.v2, axis=1))).max()

    # possible unit conversion
    if unit_conv is not None:
        max_len *= unit_conv
        vec = mesh.vectors * unit_conv
    else:
        vec = mesh.vectors
    return vec, max_len



def read_IB2d_vertices(filename):
    '''Import a Lagrangian mesh from an IB2d ascii vertex file.
    
    Parameters
    ----------
    filename : string
        path and filename of the vertex file

    Returns
    -------
    array
        Nx2 array of 2D vertices
    '''
    with open(filename) as f:
        number_of_vertices = int(f.readline())
        vertices = np.zeros((number_of_vertices,2))
        for n, line in enumerate(f):
            vertex_str = line.split()
            vertices[n,:] = (float(vertex_str[0]), float(vertex_str[1]))
    assert number_of_vertices == n+1, "Mismatch btwn stated and actual number of vertices in file."

    return vertices



@netcdf_dep
def load_netcdf(filename):
    '''Load a NetCDF file. Does not automatically read in any data.
    
    Parameters
    ----------
    filename : string
        path and filename of the NetCDF file

    Returns
    -------
    NetCDF4 Dataset object
    '''
    return nc.Dataset(filename)


#############################################################################
#                                                                           #
#                               DATA WRITERS                                #
#                                                                           #
#############################################################################


def write_vtk_point_data(path, title, data, cycle=None, time=None):
    '''Write point data to an ascii VTK file, such as agent positions. 
    
    The call 
    signature is formatted for easy looping over many time points, resulting in 
    one VTK file per time. The filename will be based on the title string and 
    the cycle number. The VTK file will be formatted as PolyData and will 
    contain field data on both CYCLE (integer time step number) and TIME (float 
    time in seconds). 
    
    Parameters
    ----------
    path : string
        directory where data should go
    title : string
        title to prepend to filename
    data : ndarray 
        position data, must be of shape Nx3
    cycle : int, optional
        dump number, e.g. integer time step in the simulation
    time : float, optional
        simulation time (generally understood to be in seconds)
    '''

    path = Path(path)
    if not path.is_dir():
        os.makedirs(path)
    if cycle is None:
        filepath = path / (title + '.vtk')
    else:
        filepath = path / (title + '_{:04d}.vtk'.format(cycle))

    vtk_data = pv.PolyData(data)
    if cycle is not None:
        vtk_data.field_data['CYCLE'] = cycle
    if time is not None:
        vtk_data.field_data['TIME'] = time
    vtk_data.save(str(filepath), binary=False)



def write_vtk_2D_rectilinear_grid_scalars(path, title, data, grid_points, 
                                          cycle=None, time=None, binary=True):
    '''Write scalar data to an ascii VTK Rectilinear Grid file (e.g. vorticity). 
    Expects data to be on a 2D rectilinear grid. Uses the pyvista library. 
    
    The call signature is formatted for easy looping over many time points, 
    resulting in one VTK file per time. The filename will be based on the title 
    string and the cycle number. The VTK file will contain field data on both 
    CYCLE (integer time step number) and TIME (float time in seconds). 
    
    Parameters
    ----------
    path : string
        directory where data should go
    title : string
        title to prepend to filename
    data : ndarray 
        scalar data, must be 2D
    grid_points : tuple (length 2) 
        grid points in each direction; Environment.flow_points
    cycle : int
        dump number, e.g. integer time step in the simulation
    time : float
        simulation time (generally understood to be in seconds)
    binary : bool
        whether or not the VTK should be binary (vs. ascii for, e.g., debugging)
    '''

    path = Path(path)
    if not path.is_dir():
        os.mkdir(path)
    if cycle is None:
        filepath = path / (title + '.vtk')
    else:
        filepath = path / (title + '_{:04d}.vtk'.format(cycle))

    # A RectilinearGrid's dimensions are implied by the coordinate arrays, so the
    # grid_points lengths must match data.shape (here (nx, ny) -> (nx, ny, 1)).
    grid = pv.RectilinearGrid(grid_points[0], grid_points[1], np.array([0.0]))
    # we have to flatten the scalar point data, and we have to do it in Fortran
    #   order because VTK moves in the x, then y, then z direction but with C
    #   memory layout, our arrays move in the z, then y, then x directions
    grid.point_data["values"] = data.flatten(order="F")
    if cycle is not None:
        grid.field_data['CYCLE'] = cycle
    if time is not None:
        grid.field_data['TIME'] = time
    grid.save(str(filepath), binary=binary)



def write_vtk_rectilinear_grid_vectors(path, title, data, grid_points, 
                                       cycle=None, time=None, binary=True):
    '''Write vector data on a 2D or 3D uniform grid (e.g. fluid velocity). Uses 
    the pyvista library. 

    The call signature is formatted for easy looping over many time points, 
    resulting in one VTK file per time. The filename will be based on the title 
    string and the cycle number. The VTK file will contain field data on both 
    CYCLE (integer time step number) and TIME (float time in seconds).

    Parameters
    ----------
    path : str
        path to a directory where the data should go. If the directory does not 
        exist, it will be created.
    title : str
        title to prepend to filenames
    data : list of ndarrays
        list of numpy array of regular grid data. The ndarrays must not include 
        a time component (e.g., the dimension of the array should match the 
        length of the list which should match the length of parameter L)
    grid_points : tuple
        grid points in each direction; Environment.flow_points
    cycle : int, optional
        dump number, which will also be included in the filename
    time : float, optional
        time stamp
    binary : bool
        whether or not the VTK should be binary (vs. ascii for, e.g., debugging)
    '''

    path = Path(path)
    if not path.is_dir():
        os.mkdir(path)
    if cycle is None:
        filepath = path / (title + '.vtk')
    else:
        filepath = path / (title + '_{:04d}.vtk'.format(cycle))

    if len(grid_points) == 2:
        # VTK data must be 3D
        grid_points = (grid_points[0], grid_points[1], np.array([0.0]))
    # A RectilinearGrid's dimensions are implied by the coordinate arrays, so the
    # grid_points lengths must match the data shape (2D data -> a singleton z).
    grid = pv.RectilinearGrid(grid_points[0], grid_points[1], grid_points[2])

    fdata = [data[ii].flatten(order='F') for ii in range(len(data))]
    if len(fdata) == 2:
        fdata += [np.zeros(fdata[0].size)]
    # something weird is going on with pyvista's support of vector data. So we 
    #   will be going a bit lower-level here.
    fdata_vtk = numpy_support.numpy_to_vtk(np.array(fdata).T)
    fdata_vtk.SetName(title)
    grid_pt_data = grid.GetPointData()
    grid_pt_data.SetVectors(fdata_vtk)
    if cycle is not None:
        grid.field_data['CYCLE'] = cycle
    if time is not None:
        grid.field_data['TIME'] = time
    grid.save(str(filepath), binary=binary)
    