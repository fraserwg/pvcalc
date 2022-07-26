from setuptools import setup
import setuptools

setup(
    name='pvcalc',
    version='0.2.0',
    author='Fraser William Goldsworth',
    author_email='fraser.goldsworth@physics.ox.ac.uk',
    packages=setuptools.find_packages(),
    #py_modules=['general'],
    scripts=[],
    url='',  #'http://pypi.python.org/pypi/PackageName/',
    license='LICENSE.txt',
    description='A package for calculating PV from MITgcm outputs',
    long_description=open('README.md').read(),
    install_requires=['xgcm'],
)
