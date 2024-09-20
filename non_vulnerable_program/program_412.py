from scipy_distutils.core      import setup
from scipy_distutils.misc_util import default_config_dict, get_path, \
     merge_config_dicts, get_subpackages

def configuration(parent_package='',parent_path=None):
    package_name = 'module'
    config = default_config_dict(package_name, parent_package)

    local_path = get_path(__name__,parent_path)
    install_path = join(*config['name'].split('.'))

    config_list = [config]

    config_list += get_subpackages(local_path,
                                   parent=config['name'],
                                   parent_path=parent_path,
                                   include_packages = ['subpackage1','subpackage2']
                                   )

    config = merge_config_dicts(config_list)

    return config

if __name__ == '__main__':
    setup(**configuration(parent_path=''))


#! /usr/bin/python
# Currently only implemented for linux...need to do win32
# I thought of making OS specific modules that were chosen 
# based on OS much like path works, but I think this will
# break pickling when process objects are passed between
# two OSes because the module name is saved in the pickle.
# A Unix process info object passed to a Windows machine
# would attempt to import the linux proc module and fail.
# Using a single module with if-then-else I think prevents
# this name problem (but it isn't as pretty...)

