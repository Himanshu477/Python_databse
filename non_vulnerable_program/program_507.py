from distutils.command.sdist import *
from distutils.command.sdist import sdist as old_sdist
from scipy.distutils.misc_util import get_data_files

class sdist(old_sdist):

    def add_defaults (self):
        old_sdist.add_defaults(self)

        if self.distribution.has_data_files():
            for data in self.distribution.data_files:
                self.filelist.extend(get_data_files(data))

        if self.distribution.has_headers():
            headers = []
            for h in self.distribution.headers:
                if isinstance(h,str): headers.append(h)
                else: headers.append(h[1])
            self.filelist.extend(headers)

        return


#!/usr/bin/python

# takes templated file .xxx.src and produces .xxx file  where .xxx is .i or .c or .h
#  using the following template rules

# /**begin repeat     on a line by itself marks the beginning of a segment of code to be repeated
# /**end repeat**/    on a line by itself marks it's end

# after the /**begin repeat and before the */
#  all the named templates are placed
#  these should all have the same number of replacements

#  in the main body, the names are used.
#  Each replace will use one entry from the list of named replacements

#  Note that all #..# forms in a block must have the same number of
#    comma-separated entries. 

__all__ = ['process_str', 'process_file']

