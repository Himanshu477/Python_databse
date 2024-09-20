    import add

The source string can be any valid Fortran code. If you want to save
the extension-module source code then a suitable file-name can be
provided by the source_fn keyword to the compile function.


Automatic extension module generation
-------------------------------------

If you want to distribute your f2py extension module, then you only
need to include the .pyf file and the Fortran code. The distutils
extensions in NumPy allow you to define an extension module entirely
in terms of this interface file. A valid setup.py file allowing
distribution of the add.f module (as part of the package f2py_examples
so that it would be loaded as f2py_examples.add) is:

.. code-block:: python

    def configuration(parent_package='', top_path=None)
        from numpy.distutils.misc_util import Configuration
        config = Configuration('f2py_examples',parent_package, top_path)
        config.add_extension('add', sources=['add.pyf','add.f'])
        return config

    if __name__ == '__main__':
