from distutils.core import setup
from distutils.extension import Extension

# Make this usable by people who don't have pyrex installed (I've committed
# the generated C sources to SVN).
try:
