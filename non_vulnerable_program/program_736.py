        import shutil
        wdir = os.path.abspath('tmp_wdir')
        if os.path.exists(wdir):
            print ' removing ',wdir
            shutil.rmtree(wdir,1)
        print ' making ',wdir
        os.mkdir(wdir)
        shutil.copy('geniotest.py',wdir)
        cwd = os.getcwd()
        os.chdir(wdir)

        run_command(sys.executable+' geniotest.py')
