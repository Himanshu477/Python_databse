import os
import sys
import string
import types
import traceback

DEBUG=0

_ppimport_is_enabled = 1
def enable():
    """ Enable postponed importing."""
    global _ppimport_is_enabled
    _ppimport_is_enabled = 1

def disable():
    """ Disable postponed importing."""
    global _ppimport_is_enabled
    _ppimport_is_enabled = 0

class PPImportError(ImportError):
    pass

def _get_so_ext(_cache={}):
    so_ext = _cache.get('so_ext')
    if so_ext is None:
        if sys.platform[:5]=='linux':
            so_ext = '.so'
        else:
            try:
                # if possible, avoid expensive get_config_vars call
                from distutils.sysconfig import get_config_vars
                so_ext = get_config_vars('SO')[0] or ''
            except ImportError:
                #XXX: implement hooks for .sl, .dll to fully support
                #     Python 1.5.x   
                so_ext = '.so'
        _cache['so_ext'] = so_ext
    return so_ext

def _get_frame(level=0):
    try:
        return sys._getframe(level+1)
    except AttributeError:
        # Python<=2.0 support
        frame = sys.exc_info()[2].tb_frame
        for i in range(level+1):
            frame = frame.f_back
        return frame

def ppimport_attr(module, name):
    """ ppimport(module, name) is 'postponed' getattr(module, name)
    """
    global _ppimport_is_enabled
    if _ppimport_is_enabled and isinstance(module, _ModuleLoader):
        return _AttrLoader(module, name)
    return getattr(module, name)

class _AttrLoader:
    def __init__(self, module, name):
        self.__dict__['_ppimport_attr_module'] = module
        self.__dict__['_ppimport_attr_name'] = name

    def _ppimport_attr_getter(self):
        module = self.__dict__['_ppimport_attr_module']
        if isinstance(module, _ModuleLoader):
            # in case pp module was loaded by other means
            module = sys.modules[module.__name__]
        attr = getattr(module,
                       self.__dict__['_ppimport_attr_name'])
        try:
            d = attr.__dict__
            if d is not None:
                self.__dict__ = d
        except AttributeError:
            pass
        self.__dict__['_ppimport_attr'] = attr

        return attr

    def __nonzero__(self):
        return 1

    def __getattr__(self, name):
        try:
            attr = self.__dict__['_ppimport_attr']
        except KeyError:
            attr = self._ppimport_attr_getter()
        if name=='_ppimport_attr':
            return attr
        return getattr(attr, name)

    def __repr__(self):
        if self.__dict__.has_key('_ppimport_attr'):
            return repr(self._ppimport_attr)
        module = self.__dict__['_ppimport_attr_module']
        name = self.__dict__['_ppimport_attr_name']
        return "<attribute %s of %s>" % (`name`,`module`)

    __str__ = __repr__

    # For function and class attributes.
    def __call__(self, *args, **kwds):
        return self._ppimport_attr(*args,**kwds)



def _is_local_module(p_dir,name,suffices):
    base = os.path.join(p_dir,name)
    for suffix in suffices:
        if os.path.isfile(base+suffix):
            if p_dir:
                return base+suffix
            return name+suffix

def ppimport(name):
    """ ppimport(name) -> module or module wrapper

    If name has been imported before, return module. Otherwise
    return ModuleLoader instance that transparently postpones
    module import until the first attempt to access module name
    attributes.
    """
    global _ppimport_is_enabled

    level = 1
    parent_frame = p_frame = _get_frame(level)
    while not p_frame.f_locals.has_key('__name__'):
        level = level + 1
        p_frame = _get_frame(level)

    p_name = p_frame.f_locals['__name__']
    if p_name=='__main__':
        p_dir = ''
        fullname = name
    elif p_frame.f_locals.has_key('__path__'):
        # python package
        p_path = p_frame.f_locals['__path__']
        p_dir = p_path[0]
        fullname = p_name + '.' + name
    else:
        # python module
        p_file = p_frame.f_locals['__file__']
        p_dir = os.path.dirname(p_file)
        fullname = p_name + '.' + name

    # module may be imported already
    module = sys.modules.get(fullname)
    if module is not None:
        if _ppimport_is_enabled or isinstance(module, types.ModuleType):
            return module
        return module._ppimport_importer()

    so_ext = _get_so_ext()
    py_exts = ('.py','.pyc','.pyo')
    so_exts = (so_ext,'module'+so_ext)

    for d,n,fn,e in [\
        # name is local python module or local extension module
        (p_dir, name, fullname, py_exts+so_exts),
        # name is local package
        (os.path.join(p_dir, name), '__init__', fullname, py_exts),
        # name is package in parent directory (scipy specific)
        (os.path.join(os.path.dirname(p_dir), name), '__init__', name, py_exts),
        ]:
        location = _is_local_module(d, n, e)
        if location is not None:
            fullname = fn
            break

    if location is None:
        # name is to be looked in python sys.path.
        fullname = name
        location = 'sys.path'

    # Try once more if module is imported.
    # This covers the case when importing from python module
    module = sys.modules.get(fullname)

    if module is not None:
        if _ppimport_is_enabled or isinstance(module,types.ModuleType):
            return module
        return module._ppimport_importer()
    # It is OK if name does not exists. The ImportError is
    # postponed until trying to use the module.

    loader = _ModuleLoader(fullname,location,p_frame=parent_frame)
    if _ppimport_is_enabled:
        return loader

    return loader._ppimport_importer()

def _get_frame_code(frame):
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    result = '%s in %s:\n' % (filename,frame.f_code.co_name)
    if not os.path.isfile(filename):
        return result
    f = open(filename)
    i = 1
    line = f.readline()
    while line:
        line = f.readline()
        i = i + 1
        if (abs(i-lineno)<2):
            result += '#%d: %s\n' % (i,line.rstrip())
        if i>lineno+3:
            break
    f.close()
    return result

def frame_traceback(frame):
    if not frame:
        return
    blocks = []
    f = frame
    while f:
        blocks.insert(0,_get_frame_code(f))
        f = f.f_back
    print '='*50
    print '\n'.join(blocks)
    print '='*50    

class _ModuleLoader:
    # Don't use it directly. Use ppimport instead.

    def __init__(self,name,location,p_frame=None):

        # set attributes, avoid calling __setattr__
        self.__dict__['__name__'] = name
        self.__dict__['__file__'] = location
        self.__dict__['_ppimport_p_frame'] = p_frame

        if location != 'sys.path':
            from scipy.base.testing import ScipyTest
            self.__dict__['test'] = ScipyTest(self).test

        # install loader
        sys.modules[name] = self

    def _ppimport_importer(self):
        name = self.__name__

        try:
            module = sys.modules[name]
    except KeyError:
        if module is not self:
            exc_info = self.__dict__.get('_ppimport_exc_info')
            if exc_info is not None:
                raise PPImportError,\
                      ''.join(traceback.format_exception(*exc_info))
            else:
                assert module is self,`(module, self)`

        # uninstall loader
        del sys.modules[name]

        if DEBUG:
            print 'Executing postponed import for %s' %(name)
        try:
            module = __import__(name,None,None,['*'])
        except Exception,msg: # ImportError:
            if DEBUG:
                p_frame = self.__dict__.get('_ppimport_p_frame',None)
                frame_traceback(p_frame)
            self.__dict__['_ppimport_exc_info'] = sys.exc_info()
            raise

        assert isinstance(module,types.ModuleType),`module`

        self.__dict__ = module.__dict__
        self.__dict__['_ppimport_module'] = module

        # XXX: Should we check the existence of module.test? Warn?
