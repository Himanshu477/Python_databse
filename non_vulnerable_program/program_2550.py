import warnings
warnings.warn("A plot_directive module is also available under "
              "matplotlib.sphinxext; expect this numpydoc.plot_directive "
              "module to be deprecated after relevant features have been "
              "integrated there.",
              FutureWarning, stacklevel=2)


#------------------------------------------------------------------------------
# Registration hook
#------------------------------------------------------------------------------

def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir
    
    app.add_config_value('plot_pre_code', '', True)
    app.add_config_value('plot_include_source', False, True)
    app.add_config_value('plot_formats', ['png', 'hires.png', 'pdf'], True)
    app.add_config_value('plot_basedir', None, True)
    app.add_config_value('plot_html_show_formats', True, True)

    app.add_directive('plot', plot_directive, True, (0, 1, False),
                      **plot_directive_options)

#------------------------------------------------------------------------------
# plot:: directive
#------------------------------------------------------------------------------
