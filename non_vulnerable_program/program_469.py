import __builtin__ as _builtin
_old_builtin_dir = _builtin.dir
def _ppimport_builtin_dir(*arg):
    if not arg:
        p_frame = _get_frame(1)
        g = p_frame.f_globals
        l = p_frame.f_locals
        l['_ppimport_old_builtin_dir'] = _old_builtin_dir
        r = eval('_ppimport_old_builtin_dir()',g,l)
        del r[r.index('_ppimport_old_builtin_dir')]
        return r
    return _old_builtin_dir(*map(_ppresolve_ignore_failure,arg))
_ppimport_builtin_dir.__doc__ = _old_builtin_dir.__doc__
_builtin.dir = _ppimport_builtin_dir


