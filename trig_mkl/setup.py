from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

setup(
    name='trig_mkl',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/n1kolasM/trig_mkl',
    license='MIT',
    author='Nikolay Marchuk',
    author_email='marchuk.nikolay.a@gmail.com',
    description='Provides python interface to Intel Math Kernel Library discrete trigonometric transform functions',
    package_data={
        'trig_mkl': ['*.pxd'],
    },
    ext_modules=cythonize(Extension('*',
                                    sources=['trig_mkl/trig_mkl.pyx'],
                                    libraries=['mkl_rt', 'mkl_avx2', 'mkl_intel_lp64', 'mkl_sequential', 'mkl_core', 'pthread', 'm', 'dl'],
                                    extra_link_args=['-Wl,--no-as-needed', '-fopenmp'],
                                    extra_compile_args=['-m64', '-fopenmp', '-O3', '-g0', '-march=native', '-mtune=native'],
                                    include_dirs=[np.get_include()]),
                          include_path=['trig_mkl']),
    install_requires=['Cython', 'numpy'],
)
