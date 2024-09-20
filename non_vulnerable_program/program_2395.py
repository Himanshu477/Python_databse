    import __main__
    __main__.__dict__.update({
        "__name__": "__main__", "__file__": "setup.py",
        "__builtins__": __builtins__})
    fp = open("setup.py", "rb")
    try:
        exec(compile(fp.read(), "setup.py", 'exec'))
    finally:
        fp.close()
except SystemExit:
    raise
except:
    traceback.print_exc()
    t = sys.exc_info()[2]
    p.interaction(None, t)
"""
            ret = subprocess.call([sys.executable, '-c', code,
                                   'build_ext', '-i'],
                                  cwd=dst,
                                  env=getenv())
            if ret != 0:
                raise RuntimeError("Build failed.")

        # Run nosetests
        subprocess.call(['nosetests3', '-v', d], cwd=TEMP)

def walk_sync(dir1, dir2, _seen=None):
    if _seen is None:
        seen = {}
    else:
        seen = _seen

    if not dir1.endswith(os.path.sep):
        dir1 = dir1 + os.path.sep

    # Walk through stuff (which we haven't yet gone through) in dir1
    for root, dirs, files in os.walk(dir1):
        sub = root[len(dir1):]
        if sub in seen:
            dirs = [x for x in dirs if x not in seen[sub][0]]
            files = [x for x in files if x not in seen[sub][1]]
            seen[sub][0].extend(dirs)
            seen[sub][1].extend(files)
        else:
            seen[sub] = (dirs, files)
        if not dirs and not files:
            continue
        yield os.path.join(dir1, sub), os.path.join(dir2, sub), dirs, files

    if _seen is None:
        # Walk through stuff (which we haven't yet gone through) in dir2
        for root2, root1, dirs, files in walk_sync(dir2, dir1, _seen=seen):
            yield root1, root2, dirs, files

def sync_2to3(src, dst, patchfile=None, clean=False):
    to_convert = []

    for src_dir, dst_dir, dirs, files in walk_sync(src, dst):
        for fn in dirs + files:
            src_fn = os.path.join(src_dir, fn)
            dst_fn = os.path.join(dst_dir, fn)

            # remove non-existing
            if os.path.exists(dst_fn) and not os.path.exists(src_fn):
                if clean:
                    if os.path.isdir(dst_fn):
                        shutil.rmtree(dst_fn)
                    else:
                        os.unlink(dst_fn)
                continue

            # make directories
            if os.path.isdir(src_fn):
                if not os.path.isdir(dst_fn):
                    os.makedirs(dst_fn)
                continue

            dst_dir = os.path.dirname(dst_fn)
            if os.path.isfile(dst_fn) and not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)

            # don't replace up-to-date files
            try:
                if os.path.isfile(dst_fn) and \
                       os.stat(dst_fn).st_mtime >= os.stat(src_fn).st_mtime:
                    continue
            except OSError:
                pass

            # copy file
            shutil.copyfile(src_fn, dst_fn)

            # add .py files to 2to3 list
            if dst_fn.endswith('.py'):
                to_convert.append(dst_fn)

    # run 2to3
    scripts = []

    for fn in list(to_convert):
        if os.path.basename(fn) in SCRIPTS:
            scripts.append(fn)
            to_convert.remove(fn)

    if patchfile:
        p = open(patchfile, 'wb+')
    else:
        p = open(os.devnull, 'wb')

    if to_convert:
        subprocess.call(['2to3', '-w'] + to_convert, stdout=p)
    if scripts:
        subprocess.call(['2to3', '-w', '-x', 'import'] + scripts, stdout=p)
    p.close()

if __name__ == "__main__":
    main()


"""
Python 3 compatibility tools.

"""

__all__ = ['bytes', 'asbytes', 'isfile']

