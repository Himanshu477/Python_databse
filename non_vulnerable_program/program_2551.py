from docutils.parsers.rst import directives
from docutils import nodes

def plot_directive(name, arguments, options, content, lineno,
                   content_offset, block_text, state, state_machine):
    return run(arguments, content, options, state_machine, state, lineno)
plot_directive.__doc__ = __doc__

def _option_boolean(arg):
    if not arg or not arg.strip():
        # no argument given, assume used as a flag
        return True
    elif arg.strip().lower() in ('no', '0', 'false'):
        return False
    elif arg.strip().lower() in ('yes', '1', 'true'):
        return True
    else:
        raise ValueError('"%s" unknown boolean' % arg)

def _option_format(arg):
    return directives.choice(arg, ('python', 'lisp'))

def _option_align(arg):
    return directives.choice(arg, ("top", "middle", "bottom", "left", "center",
                                   "right"))

plot_directive_options = {'alt': directives.unchanged,
                          'height': directives.length_or_unitless,
                          'width': directives.length_or_percentage_or_unitless,
                          'scale': directives.nonnegative_int,
                          'align': _option_align,
                          'class': directives.class_option,
                          'include-source': _option_boolean,
                          'format': _option_format,
                          }

#------------------------------------------------------------------------------
# Generating output
#------------------------------------------------------------------------------

