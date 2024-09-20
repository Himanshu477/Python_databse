    import pkg_resources # activate namespace packages (manipulates __path__)
except ImportError:
    pass

pkgload = PackageLoader()

if show_core_config is None:
    print >> sys.stderr, 'Running from numpy core source directory.'
else:
