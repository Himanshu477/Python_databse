from function_base import any, all

def _import_fail_message(module, version):
    """Prints a message when the array package specific version of an extension
    fails to import correctly.
    """
    _dict = { "which" : which[0],
              "module" : module,
              "specific" : version + module
              }
    print """\nThe import of the %(which)s version of the %(module)s module, %(specific)s, failed.\nThis is either because %(which)s was unavailable when scipy was compiled,\nor because a dependency of %(specific)s could not be satisfied.\nIf it appears that %(specific)s was not built,  make sure you have a working copy of\n%(which)s and then re-install scipy. Otherwise, the following traceback gives more details:\n""" % _dict


# http://g95.sourceforge.net/

