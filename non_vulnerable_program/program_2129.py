import os
import tempfile
from shutil import rmtree
from urllib2 import urlopen, URLError
from urlparse import urlparse

# TODO: .zip support, .tar support?
_file_openers = {None: open}
try:
