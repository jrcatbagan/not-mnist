#!/usr/bin/python
# Incorporate compatibility between python version 2 and version 3
from __future__ import print_function
# Provide a MATLAB-like plotting framework
# from matplotlib import pyplot
# Gain access to the fundamental package for scientific computing within python
# import numpy
# Gain access to URL utilities
from six.moves.urllib.request import urlretrieve
import os
import tarfile

data_root = '.'
filename_base = 'notMNIST_large'
url = 'https://commondatastorage.googleapis.com/books1000/'
dataset_archived_file = filename_base + '.tar.gz'

print("Extracting the archived dataset file")

try:
    tar = tarfile.open(dataset_archived_file, 'r:gz')
    tar.extractall(data_root)
    tar.close()
except OSError:
    print("An error has occured")
