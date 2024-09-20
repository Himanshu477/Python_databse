import numpy as np
major, minor = [ int(d) for d in np.__version__.split(".")[:2] ]
if major == 0: BadListError = TypeError
else:          BadListError = ValueError

