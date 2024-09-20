        from numpy.distutils.exec_command import exec_command
        sts = exec_command(setup_cmd)
        #p = subprocess.Popen(setup_cmd, cwd=build_dir, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        #sts = os.waitpid(p.pid, 0)
        if sts[0]:
            raise "Failed to build (status=%s)." % (`sts`)
        exec 'import %s as m' % (modulename)
        return m

def _test():
    import doctest
    doctest.testmod()
    
if __name__ == "__main__":
    _test()



