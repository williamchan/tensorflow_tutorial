"""Functions for downloading and reading MNIST data."""

# These 3 lines provides backward compatibility with older Python versions from Python 3 code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# six is a package that helps in writing code that is compatible with both Python 2 and Python 3.
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import gzip
import os
import tempfile
import numpy
import tensorflow as tf

# The mnist read_data_sets() function will be used in full_connected_feed.py to download mnist dataset
# to your local training folder and to then unpack that data to return a dictionary of DataSet instances.
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
