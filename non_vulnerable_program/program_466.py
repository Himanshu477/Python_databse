    import pydoc as _pydoc
except ImportError:
    _pydoc = None

if _pydoc is not None:
    # Redefine __call__ method of help.__class__ to
    # support ppimport.
