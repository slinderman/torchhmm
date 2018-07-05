#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='torchhmm',
      version='0.0.1',
      description='Gradients through HMM message passing',
      author='Scott Linderman',
      author_email='scott.linderman@gmail.com',
      url='http://www.github.com/slinderman/torchhmm',
      packages=['torchhmm'],
      ext_modules=cythonize('**/*.pyx'),
      include_dirs=[np.get_include(),],
      )
