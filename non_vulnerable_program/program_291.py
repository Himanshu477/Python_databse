from scipy_distutils import scipy_distutils_version as v1
from scipy_test import scipy_test_version as v2

major = max(v1.major,v2.major)
minor = max(v1.minor,v2.minor)
micro = max(v1.micro,v2.micro)
release_level = min(v1.release_level,v2.release_level)
cvs_minor = v1.cvs_minor + v2.cvs_minor
cvs_serial = v1.cvs_serial + v2.cvs_serial

scipy_core_version = '%(major)d.%(minor)d.%(micro)d_%(release_level)s'\
                     '_%(cvs_minor)d.%(cvs_serial)d' % (locals ())

if __name__ == "__main__":
