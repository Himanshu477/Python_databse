        from scipy.base.testing import ScipyTest
        module.test = ScipyTest(module).test

        return module

    def __setattr__(self, name, value):
        try:
            module = self.__dict__['_ppimport_module']
        except KeyError:
            module = self._ppimport_importer()
        return setattr(module, name, value)

    def __getattr__(self, name):
        try:
            module = self.__dict__['_ppimport_module']
        except KeyError:
            module = self._ppimport_importer()
        return getattr(module, name)

    def __repr__(self):
        global _ppimport_is_enabled
        if not _ppimport_is_enabled:
            try:
                module = self.__dict__['_ppimport_module']
            except KeyError:
                module = self._ppimport_importer()
            return module.__repr__()
        if self.__dict__.has_key('_ppimport_module'):
            status = 'imported'
        elif self.__dict__.has_key('_ppimport_exc_info'):
            status = 'import error'
        else:
            status = 'import postponed'
        return '<module %s from %s [%s]>' \
               % (`self.__name__`,`self.__file__`, status)

    __str__ = __repr__

def ppresolve(a,ignore_failure=None):
    """ Return resolved object a.

    a can be module name, postponed module, postponed modules
    attribute, string representing module attribute, or any
    Python object.
    """
    global _ppimport_is_enabled
    if _ppimport_is_enabled:
        disable()
        a = ppresolve(a,ignore_failure=ignore_failure)
        enable()
        return a
    if type(a) is type(''):
        ns = a.split('.')
        if ignore_failure:
            try:
                a = ppimport(ns[0])
            except:
                return a
        else:
            a = ppimport(ns[0])
        b = [ns[0]]
        del ns[0]
        while ns:
            if hasattr(a,'_ppimport_importer') or \
                   hasattr(a,'_ppimport_module'):
                a = getattr(a,'_ppimport_module',a)
            if hasattr(a,'_ppimport_attr'):
                a = a._ppimport_attr
            b.append(ns[0])
            del ns[0]
            if ignore_failure and not hasattr(a, b[-1]):
                a = '.'.join(ns+b)
                b = '.'.join(b)
                if sys.modules.has_key(b) and sys.modules[b] is None:
                    del sys.modules[b]
                return a
            a = getattr(a,b[-1])
    if hasattr(a,'_ppimport_importer') or \
           hasattr(a,'_ppimport_module'):
        a = getattr(a,'_ppimport_module',a)
    if hasattr(a,'_ppimport_attr'):
        a = a._ppimport_attr
    return a

def _ppresolve_ignore_failure(a):
    return ppresolve(a,ignore_failure=1)

try:
