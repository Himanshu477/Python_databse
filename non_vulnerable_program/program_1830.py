    from __svn_version__ import version
    version_info = (major, version)
    version = '%s_%s' % version_info
except ImportError:
    version = str(major)


#! /usr/bin/env python

# System imports
