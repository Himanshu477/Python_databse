import bz2
import gzip
import os
import sys
import struct
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile
from shutil import rmtree
from urlparse import urlparse

