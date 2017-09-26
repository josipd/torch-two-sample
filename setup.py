from setuptools import setup, Extension
import numpy


setup(
    name='torch-two-sample',
    version='0.1',
    description='A PyTorch library for (differentiable) two sample tests',
    author='Josip Djolonga',
    author_email='josipd@inf.ethz.ch',
    setup_requires=[
        'setuptools>=18.0',
        'cython',
        'pytest-runner',
    ],
    install_requires=[
        'torch',
        'numpy',
        'scipy',
    ],
    tests_require=[
        'pytest',
    ],
    packages=['torch_two_sample'],
    ext_modules=[
        Extension('torch_two_sample.permutation_test',
                  sources=['torch_two_sample/permutation_test.pyx'],
                  include_dirs=[numpy.get_include()]),
    ],
    license='license.txt',
)
