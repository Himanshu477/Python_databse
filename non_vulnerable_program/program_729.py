from __version__ import version

class my_install_data (install_data):
    def finalize_options (self):
        self.set_undefined_options ('install',
                                    ('install_lib', 'install_dir'),
                                    ('root', 'root'),
                                    ('force', 'force'),
                                    )

def f2py_py():
    return '''#!/usr/bin/env %s
# See http://cens.ioc.ee/projects/f2py2e/
