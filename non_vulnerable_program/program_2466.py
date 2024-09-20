import util

def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))

