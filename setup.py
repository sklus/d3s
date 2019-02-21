#!env python3
from setuptools import setup, find_packages, Extension

import os
import sys

boost = os.getenv('BOOST', False)

if not boost:
    print('please set env variable "BOOST"  to the location of your boost library')
    sys.exit(1)

if not os.path.exists(boost):
    print('boost does not exist')
    sys.exit(2)


e = Extension('dspy.systems',
              sources=['cpp/systems.cpp'],
              language='c++',
              extra_compile_args=['-c', '-O3', '-fPIC', '-D_UNIX', '-std=c++11', '-Wno-deprecated-declarations'],
              include_dirs=[boost, 'm'],
              libraries=['boost_python', 'boost_numpy3'],
              )


setup(name='dspy',
      version='0.1',
      ext_modules=[e],
      packages=find_packages(),
      include_package_data=True,
)
