import os
import sys
import imp
import types
import unittest
import traceback

__all__ = ['set_package_path', 'set_local_path', 'restore_path',
           'IgnoreException', 'ScipyTestCase', 'ScipyTest']

DEBUG=0
get_frame = sys._getframe
