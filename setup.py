from setuptools import setup

setup(
   name='exacthgm-lauziere',
   version='1.0',
   description='Python3 implementation of "An Exact Hypergraph Matching Algorithm for Nuclear Identification in Embryonic C. elegans"',
   author='Andrew Lauziere',
   author_email='lauziere@umd.edu',
   packages=['exacthgm'],  
   license='MIT',
   python_requires='>=3.6',
   url="https://github.com/lauziere/Exact_HGM",
   install_requires=['numpy>=1.19.5', 'pandas>=1.1.5', 'scipy>=1.5.4']
)