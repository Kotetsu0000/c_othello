'''
pip install cython numpy
pip install .
python setup.py build_ext --inplace
'''
from setuptools import setup, Extension, find_packages
from Cython.Build import build_ext, cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
Cython.Compiler.Options.docstrings = True
Cython.Compiler.Options.embed_pos_in_docstring = True
import numpy


setup_kwargs = {
    'name': 'cython_othello',
    'version': '0.0.3',
    'description': 'Cython implementation of Othello game',
    'author': 'Kotetsu0000',
    'ext_modules': cythonize(
        [
            Extension('c_othello.c_othello', ['c_othello/c_othello.pyx'], language='c++', extra_compile_args=['/openmp'], extra_link_args=['/openmp']),
        ],
        compiler_directives={'language_level': 3, 'embedsignature': True},
    ),
    'include_dirs': [numpy.get_include()],
    'install_requires': ['numpy'],
}

setup(**setup_kwargs)

