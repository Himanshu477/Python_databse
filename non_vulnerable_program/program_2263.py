    from docutils.parsers.rst.directives import _directives

    def html_only_directive(name, arguments, options, content, lineno,
                            content_offset, block_text, state, state_machine):
        return run(content, html_only, state, content_offset)

    def latex_only_directive(name, arguments, options, content, lineno,
                             content_offset, block_text, state, state_machine):
        return run(content, latex_only, state, content_offset)

    for func in (html_only_directive, latex_only_directive):
        func.content = 1
        func.options = {}
        func.arguments = None

    _directives['htmlonly'] = html_only_directive
    _directives['latexonly'] = latex_only_directive
else:
    class OnlyDirective(Directive):
        has_content = True
        required_arguments = 0
        optional_arguments = 0
        final_argument_whitespace = True
        option_spec = {}

        def run(self):
            self.assert_has_content()
            return run(self.content, self.node_class,
                       self.state, self.content_offset)

    class HtmlOnlyDirective(OnlyDirective):
        node_class = html_only

    class LatexOnlyDirective(OnlyDirective):
        node_class = latex_only

    directives.register_directive('htmlonly', HtmlOnlyDirective)
    directives.register_directive('latexonly', LatexOnlyDirective)

def setup(app):
    app.add_node(html_only)
    app.add_node(latex_only)

    # Add visit/depart methods to HTML-Translator:
    def visit_perform(self, node):
        pass
    def depart_perform(self, node):
        pass
    def visit_ignore(self, node):
        node.children = []
    def depart_ignore(self, node):
        node.children = []

    HTMLTranslator.visit_html_only = visit_perform
    HTMLTranslator.depart_html_only = depart_perform
    HTMLTranslator.visit_latex_only = visit_ignore
    HTMLTranslator.depart_latex_only = depart_ignore

    LaTeXTranslator.visit_html_only = visit_ignore
    LaTeXTranslator.depart_html_only = depart_ignore
    LaTeXTranslator.visit_latex_only = visit_perform
    LaTeXTranslator.depart_latex_only = depart_perform


"""
==============
phantom_import
==============

Sphinx extension to make directives from ``sphinx.ext.autodoc`` and similar
extensions to use docstrings loaded from an XML file.

This extension loads an XML file in the Pydocweb format [1] and
creates a dummy module that contains the specified docstrings. This
can be used to get the current docstrings from a Pydocweb instance
without needing to rebuild the documented module.

.. [1] http://code.google.com/p/pydocweb

"""
