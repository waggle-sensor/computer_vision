from distutils.core import setup
from Cython.Build import cythonize

"""
This function just just responsible for compiling cython files.
"""

setup(
    ext_modules = cythonize(["grad_hist.pyx", "slide_window.pyx", "cascade.pyx"])
)