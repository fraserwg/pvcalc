from setuptools import setup
import setuptools

setup(
    name='PVCALC',
    version='0.0.1',
    author='Fraser William Goldsworth',
    author_email='fraser.goldsworth@physics.ox.ac.uk',
    packages=setuptools.find_packages(),
    #py_modules=['general'],
    scripts=['scripts/level_script.py', 'scripts/slice_script.py'],
    url='',  #'http://pypi.python.org/pypi/PackageName/',
    license='LICENSE.txt',
    description='A package for calculating PV from MITgcm outputs',
    long_description=open('README.md').read(),
    install_requires=[],
)
