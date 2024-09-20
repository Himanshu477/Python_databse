from parser.api import parse, PythonModule, EndStatement

def dump_signature():
    """ Read Fortran files and dump the signatures to file or stdout.
    """
    # get signature output
    i = sys.argv.index('-h')
    if len(sys.argv)-1==i:
        signature_output = 'stdout'
    else:
        signature_output = sys.argv[i+1]
        del sys.argv[i+1]
    del sys.argv[i]

    # get module name
    try:
        i = sys.argv.index('-m')
    except ValueError:
        i = None
        module_name = 'unknown'
    if i is not None:
        if len(sys.argv)-1==i:
            module_name = 'unspecified'
        else:
            module_name = sys.argv[i+1]
            del sys.argv[i+1]
        del sys.argv[i]

    # initialize output stream
    if signature_output in ['stdout','stderr']:
        output_stream = getattr(sys, signature_output)
    else:
        if os.path.isfile(signature_output):
            try:
                i = sys.argv.index('--overwrite-signature')
                del sys.argv[i]
            except ValueError:
                print >> sys.stderr, 'Signature file %r exists. Use --overwrite-signature to overwrite.' % (signature_output)
                sys.exit()
        output_stream = open(signature_output,'w')

    flag = 'file'
    file_names = []
    only_names = []
    skip_names = []
    options = []
    for word in sys.argv[1:]:
        if word=='': pass
        elif word=='only:': flag = 'only'
        elif word=='skip:': flag = 'skip'
        elif word==':': flag = 'file'
        elif word.startswith('--'): options.append(word)
        else:
            {'file': file_names,'only': only_names, 'skip': skip_names}[flag].append(word)

    output_stream.write('''!    -*- f90 -*-
! Note: the context of this file is case sensitive.
''')
    output_stream.write('PYTHON MODULE %s\n' % (module_name))
    output_stream.write('  INTERFACE\n\n')
    for filename in file_names:
        block = parse(filename)
        output_stream.write('! File: %s, source mode = %s\n' % (filename, block.reader.mode))
        if isinstance(block.content[0],PythonModule):
            for subblock in block.content[0].content[0].content:
                if isinstance(subblock, EndStatement):
                    break
                output_stream.write(subblock.topyf('    ')+'\n')
        else:
            output_stream.write(block.topyf('    ')+'\n')
    output_stream.write('  END INTERFACE\n')
    output_stream.write('END PYTHON MODULE %s\n' % (module_name))
    
    if signature_output not in ['stdout','stderr']:
        output_stream.close()
    return

def build_extension():
    raise NotImplementedError,'build_extension'

def main():
    """ Main function of f2py script.
    """
    if '--help-link' in sys.argv[1:]:
        sys.argv.remove('--help-link')
        from numpy.distutils.system_info import show_all
        show_all()
        return
    if '-c' in sys.argv[1:]:
        i = sys.argv.index('-c')
        del sys.argv[i]
        build_extension()
        return
    if '-h' in sys.argv[1:]:
        dump_signature()
        return
    print >> sys.stdout, __usage__


#!/usr/bin/env python
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('lib',parent_package,top_path)
    config.add_subpackage('parser')
    config.add_data_files('*.txt','parser/*.txt')
    config.add_data_dir('src')
    return config

if __name__ == "__main__":
