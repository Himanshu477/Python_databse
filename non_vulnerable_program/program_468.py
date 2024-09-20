import inspect as _inspect
_old_inspect_getfile = _inspect.getfile
def _ppimport_inspect_getfile(object):
    if isinstance(object,_ModuleLoader):
        return object.__dict__['__file__']
    return _old_inspect_getfile(_ppresolve_ignore_failure(object))
_ppimport_inspect_getfile.__doc__ = _old_inspect_getfile.__doc__
_inspect.getfile = _ppimport_inspect_getfile

_old_inspect_getdoc = _inspect.getdoc
def _ppimport_inspect_getdoc(object):
    return _old_inspect_getdoc(_ppresolve_ignore_failure(object))
_ppimport_inspect_getdoc.__doc__ = _old_inspect_getdoc.__doc__
_inspect.getdoc = _ppimport_inspect_getdoc

_old_inspect_getsource = _inspect.getsource
def _ppimport_inspect_getsource(object):
    return _old_inspect_getsource(_ppresolve_ignore_failure(object))
_ppimport_inspect_getsource.__doc__ = _old_inspect_getsource.__doc__
_inspect.getsource = _ppimport_inspect_getsource

