'''setup.py file

Type "python setup.py install" into a terminal to install the package for basic
usage.

Alternatively, type "python setup.py install --inplace" to create an installation 
that will read from the current directory.
'''

from setuptools import setup

install_requires = ['numpy',
                    'scipy',
                    'matplotlib',
                    'pandas',
                    'ffmpeg>=4.3.1'
                    ]

full_install_requires = ['numpy',
                        'scipy',
                        'matplotlib>=3.0.0',
                        'pandas',
                        'ffmpeg>=4.3.1',
                        'vtk',
                        'pyvista',
                        'numpy-stl',
                        'pytest'
                        ]

setup(
    name='Planktos',
    version='0.3',
    description='Planktos Agent-based Modeling Framework',
    author='Christopher Strickland',
    author_email='cstric12@utk.edu',
    license='GPLv3',
    url='https://github.com/mountaindust/Planktos',
    keywords='ABM CFD FTLE agents',
    python_requires='>=3.5',
    packages=['planktos'],
    install_requires=install_requires,
)