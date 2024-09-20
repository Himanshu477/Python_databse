    import new as _new

    _old_pydoc_help_call = _pydoc.help.__class__.__call__
    def _ppimport_pydoc_help_call(self,*args,**kwds):
        return _old_pydoc_help_call(self, *map(_ppresolve_ignore_failure,args),
                                    **kwds)
    _ppimport_pydoc_help_call.__doc__ = _old_pydoc_help_call.__doc__
    _pydoc.help.__class__.__call__ = _new.instancemethod(_ppimport_pydoc_help_call,
                                                         None,
                                                         _pydoc.help.__class__)

    _old_pydoc_Doc_document = _pydoc.Doc.document
    def _ppimport_pydoc_Doc_document(self,*args,**kwds):
        args = (_ppresolve_ignore_failure(args[0]),) + args[1:]
        return _old_pydoc_Doc_document(self,*args,**kwds)
    _ppimport_pydoc_Doc_document.__doc__ = _old_pydoc_Doc_document.__doc__
    _pydoc.Doc.document = _new.instancemethod(_ppimport_pydoc_Doc_document,
                                              None,
                                              _pydoc.Doc)

    _old_pydoc_describe = _pydoc.describe
    def _ppimport_pydoc_describe(object):
        return _old_pydoc_describe(_ppresolve_ignore_failure(object))
    _ppimport_pydoc_describe.__doc__ = _old_pydoc_describe.__doc__
    _pydoc.describe = _ppimport_pydoc_describe

