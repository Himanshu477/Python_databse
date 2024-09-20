        from scipy_distutils.core import setup,Extension
        ext = Extension('iotest',pyf_sources,f2py_options = f2py_opts)

        mess(' running setup..\n')
        setup(ext_modules = [ext])
        #####################################################

        sys.argv = sys_argv
        mess(' running tests..')
        status,output=run_command(sys.executable + ' runiotest.py')
        if status:
            mess('failed\n')
        else:
            succ,fail = string.count(output,'SUCCESS'),string.count(output,'FAILURE')
            mess('%s passed, %s failed\n'%(succ,fail))
            if fail:
