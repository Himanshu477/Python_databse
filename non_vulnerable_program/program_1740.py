import sys
import os
import re
import glob


_args3 = ['compress', 'take', 'repeat']
_funcm1 = ['nansum', 'nanmax', 'nanmin', 'nanargmax', 'nanargmin',
           'argmax', 'argmin', 'compress']
_func0 = ['take', 'repeat', 'sum', 'product', 'sometrue', 'alltrue',
          'cumsum', 'cumproduct', 'average', 'ptp', 'cumprod', 'prod',
          'std', 'mean']

_all = _func0 + _funcm1
func_re = {}

for name in _all:
    _astr = r"""%s\s*[(]"""%name 
    func_re[name] = re.compile(_astr)


