import os
import string
import sys
import re
from glob import glob
from types import *
from distutils.command.build_clib import build_clib as old_build_clib
from distutils.command.build_clib import show_compilers

