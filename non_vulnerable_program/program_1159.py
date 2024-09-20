    import imp
    svn = imp.load_module('numpy.base.__svn_version__',
                          open(svn_version_file),
                          svn_version_file,
                          ('.py','U',1))
    version += '.'+svn.version


# To get sub-modules
