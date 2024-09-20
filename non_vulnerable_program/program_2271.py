    from docutils.parsers.rst.directives import _directives

    def plot_directive(name, arguments, options, content, lineno,
                       content_offset, block_text, state, state_machine):
        return run(arguments, options, state_machine, lineno)
    plot_directive.__doc__ = __doc__
    plot_directive.arguments = (1, 0, 1)
    plot_directive.options = options

    _directives['plot'] = plot_directive
else:
    class plot_directive(Directive):
        required_arguments = 1
        optional_arguments = 0
        final_argument_whitespace = True
        option_spec = options
        def run(self):
            return run(self.arguments, self.options,
                       self.state_machine, self.lineno)
    plot_directive.__doc__ = __doc__

    directives.register_directive('plot', plot_directive)

def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir

    app.add_config_value('plot_output_dir', '_static', True)
    app.add_config_value('plot_pre_code', '', True)

plot_directive.__doc__ = __doc__

directives.register_directive('plot', plot_directive)



# -*- encoding:utf-8 -*-

