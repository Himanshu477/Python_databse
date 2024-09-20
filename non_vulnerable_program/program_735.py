    import commands
    run_command = commands.getstatusoutput


def main():
    """
    Ensure that code between try: ... finally: is executed with
    exactly the same conditions regardless of the way how this
    function is called (as a script from an arbitrary path or
    as a module member function, but not through execfile function).
    """
    sys_stdout = sys.stdout
    sys_stderr = sys.stderr
    def mess(text,sys_stdout=sys_stdout):
        sys.stdout.write(text)
        sys_stdout.write(text)
    _log = open(os.path.abspath(__file__)+'.log','w')
    sys.stdout = sys.stderr = _log
    _sys_argv = sys.argv
    mess('Running %s\n'%(`string.join(sys.argv,' ')`))
    mess(' log is saved to %s\n'%(_log.name))
    _path = os.path.abspath(os.path.dirname(__file__))
    _f2pypath = os.path.normpath(os.path.join(_path,'..','..'))
    sys.path.insert(0,_f2pypath)
    _cwd = os.getcwd()
    os.chdir(_path)

    
    try:
        ############## CODE TO BE TESTED #################

