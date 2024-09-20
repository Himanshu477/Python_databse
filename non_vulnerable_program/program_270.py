import os,sys,time,glob,string,traceback,unittest
import fastumath as math

try:
    # These are used by Numeric tests.
    # If Numeric and scipy_base  are not available, then some of the
    # functions below will not be available.
