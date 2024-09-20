                import re
                res = re.findall(r'FAILURE.*\n',output)
                mess(string.join(res,''))
        print 30*'*'+' TEST OUTPUT '+30*'*'
        print output
        print 30*'*'+' END OF TEST OUTPUT '+30*'*'            
        os.chdir(cwd)

        ############## END OF CODE TO BE TESTED #################
    finally:
        os.chdir(_cwd)
        del sys.path[0]
        sys.stdout = sys_stdout
        sys.stderr = sys_stderr
        sys.argv = _sys_argv
        _log.close()


if __name__ == "__main__":
    main()


#!/usr/bin/env python

