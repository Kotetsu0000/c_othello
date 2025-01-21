'''
pip install cython numpy
pip install .
python setup.py build_ext --inplace
'''
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
Cython.Compiler.Options.docstrings = True
Cython.Compiler.Options.embed_pos_in_docstring = True

from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy


'''setup_kwargs = {
    'name': 'cython_othello',
    'version': '0.0.4',
    'description': 'Cython implementation of Othello game',
    'author': 'Kotetsu0000',
    'ext_modules': cythonize(
        [
            Extension('c_othello.c_othello', ['c_othello/c_othello.pyx'], language='c++'),
        ],
        compiler_directives={'language_level': 3, 'embedsignature': True},
    ),
    'include_dirs': [numpy.get_include()],
    'packages': find_packages(),
    'install_requires': ['numpy'],
}'''

setup_kwargs = {
    'name': 'cython_othello',
    'version': '0.0.4',
    'description': 'Cython implementation of Othello game',
    'author': 'Kotetsu0000',
    'ext_modules': [
        Pybind11Extension('c_othello.c_othello_bit', ['c_othello/c_othello_bit.cpp'], language='c++'),
        Extension('c_othello.c_othello', ['c_othello/c_othello.cpp'], language='c++'),
    ],
    'include_dirs': [numpy.get_include()],
    'packages': find_packages(),
    'install_requires': ['numpy'],
}

setup(**setup_kwargs)

