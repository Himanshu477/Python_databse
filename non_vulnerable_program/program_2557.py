import exceptions

def contains_doctest(text):
    try:
        # check if it's valid Python as-is
        compile(text, '<string>', 'exec')
        return False
    except SyntaxError:
        pass
    r = re.compile(r'^\s*>>>', re.M)
    m = r.search(text)
    return bool(m)

def unescape_doctest(text):
    """
    Extract code from a piece of text, which contains either Python code
    or doctests.

    """
    if not contains_doctest(text):
        return text

    code = ""
    for line in text.split("\n"):
        m = re.match(r'^\s*(>>>|\.\.\.) (.*)$', line)
        if m:
            code += m.group(2) + "\n"
        elif line.strip():
            code += "# " + line.strip() + "\n"
        else:
            code += "\n"
    return code

def split_code_at_show(text):
    """
    Split code at plt.show()

    """

    parts = []
    is_doctest = contains_doctest(text)

    part = []
    for line in text.split("\n"):
        if (not is_doctest and line.strip() == 'plt.show()') or \
               (is_doctest and line.strip() == '>>> plt.show()'):
            part.append(line)
            parts.append("\n".join(part))
            part = []
        else:
            part.append(line)
    if "\n".join(part).strip():
        parts.append("\n".join(part))
    return parts

class PlotError(RuntimeError):
    pass

def run_code(code, code_path, ns=None):
    # Change the working directory to the directory of the example, so
    # it can get at its data files, if any.
    pwd = os.getcwd()
    old_sys_path = list(sys.path)
    if code_path is not None:
        dirname = os.path.abspath(os.path.dirname(code_path))
        os.chdir(dirname)
        sys.path.insert(0, dirname)

    # Redirect stdout
    stdout = sys.stdout
    sys.stdout = StringIO()

    # Reset sys.argv
    old_sys_argv = sys.argv
    sys.argv = [code_path]
    
    try:
        try:
            code = unescape_doctest(code)
            if ns is None:
                ns = {}
            if not ns:
                exec(setup.config.plot_pre_code, ns)
            exec(code, ns)
        except (Exception, SystemExit) as err:
            raise PlotError(traceback.format_exc())
    finally:
        os.chdir(pwd)
        sys.argv = old_sys_argv
        sys.path[:] = old_sys_path
        sys.stdout = stdout
    return ns


#------------------------------------------------------------------------------
# Generating figures
#------------------------------------------------------------------------------

def out_of_date(original, derived):
    """
    Returns True if derivative is out-of-date wrt original,
    both of which are full file paths.
    """
    return (not os.path.exists(derived)
            or os.stat(derived).st_mtime < os.stat(original).st_mtime)


def makefig(code, code_path, output_dir, output_base, config):
    """
    Run a pyplot script *code* and save the images under *output_dir*
    with file names derived from *output_base*

    """

    # -- Parse format list
    default_dpi = {'png': 80, 'hires.png': 200, 'pdf': 50}
    formats = []
    for fmt in config.plot_formats:
        if isinstance(fmt, str):
            formats.append((fmt, default_dpi.get(fmt, 80)))
        elif type(fmt) in (tuple, list) and len(fmt)==2:
            formats.append((str(fmt[0]), int(fmt[1])))
        else:
            raise PlotError('invalid image format "%r" in plot_formats' % fmt)

    # -- Try to determine if all images already exist

    code_pieces = split_code_at_show(code)

    # Look for single-figure output files first
    all_exists = True
    img = ImageFile(output_base, output_dir)
    for format, dpi in formats:
        if out_of_date(code_path, img.filename(format)):
            all_exists = False
            break
        img.formats.append(format)

    if all_exists:
        return [(code, [img])]

    # Then look for multi-figure output files
    results = []
    all_exists = True
    for i, code_piece in enumerate(code_pieces):
        images = []
        for j in range(1000):
            img = ImageFile('%s_%02d_%02d' % (output_base, i, j), output_dir)
            for format, dpi in formats:
                if out_of_date(code_path, img.filename(format)):
                    all_exists = False
                    break
                img.formats.append(format)

            # assume that if we have one, we have them all
            if not all_exists:
                all_exists = (j > 0)
                break
            images.append(img)
        if not all_exists:
            break
        results.append((code_piece, images))

    if all_exists:
        return results

    # -- We didn't find the files, so build them

    results = []
    ns = {}

    for i, code_piece in enumerate(code_pieces):
        # Clear between runs
        plt.close('all')

        # Run code
        run_code(code_piece, code_path, ns)

        # Collect images
        images = []
        fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
        for j, figman in enumerate(fig_managers):
            if len(fig_managers) == 1 and len(code_pieces) == 1:
                img = ImageFile(output_base, output_dir)
            else:
                img = ImageFile("%s_%02d_%02d" % (output_base, i, j),
                                output_dir)
            images.append(img)
            for format, dpi in formats:
                try:
                    figman.canvas.figure.savefig(img.filename(format), dpi=dpi)
                except exceptions.BaseException as err:
                    raise PlotError(traceback.format_exc())
                img.formats.append(format)

        # Results
        results.append((code_piece, images))

    return results


#------------------------------------------------------------------------------
# Relative pathnames
#------------------------------------------------------------------------------

try:
