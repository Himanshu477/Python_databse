from docutils.nodes import Body, Element
from docutils.writers.html4css1 import HTMLTranslator
from sphinx.latexwriter import LaTeXTranslator
from docutils.parsers.rst import directives

class html_only(Body, Element):
    pass

class latex_only(Body, Element):
    pass

def run(content, node_class, state, content_offset):
    text = '\n'.join(content)
    node = node_class(text)
    state.nested_parse(content, content_offset, node)
    return [node]

try:
