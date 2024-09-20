from base import Component

class Word(Component):
    template = '%(word)s'

    def initialize(self, word):
        if not word: return None
        self.word = word
        return self

    def add(self, component, container_label=None):
        raise ValueError('%s does not take components' % (self.__class__.__name__))

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,[self.word]+[c for (c,l) in self.components])))


class Line(Component):

    """
    >>> l = Line('hey')
    >>> l += ' you '
    >>> l += 2
    >>> print l
    Line('hey you 2')
    >>> print l.generate()
    hey you 2
    >>> l += l
    >>> print l.generate()
    hey you 2hey you 2
    """

    template = '%(line)s'

    def initialize(self, *strings):
        self.line = ''
        map(self.add, strings)
        return self

    def add(self, component, container_label=None):
        if isinstance(component, Line):
            self.line += component.line
        elif isinstance(component, str):
            self.line += component
        elif component is None:
            pass
        else:
            self.line += str(component)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,[self.line]+[c for (c,l) in self.components])))


class Code(Component):

    """
    >>> c = Code('start')
    >>> c += 2
    >>> c += 'end'
    >>> c
    Code(Line('start'), Line('2'), Line('end'))
    >>> print c.generate()
    start
    2
    end
    """

    template = '%(Line)s'

    container_options = dict(
        Line = dict(default = '<KILLLINE>', ignore_empty_content=True)
        )
    component_container_map = dict(
        Line = 'Line'
        )
    default_component_class_name = 'Line'

    def initialize(self, *lines):
        map(self.add, lines)
        return self

    def add(self, component, label=None):
        if isinstance(component, Code):
            assert label is None,`label`
            self.components += component.components
        else:
            Component.add(self, component, label)


class FileSource(Component):

    container_options = dict(
        Content = dict(default='<KILLLINE>')
        )

    template = '%(Content)s'

    default_component_class_name = 'Code'

    component_container_map = dict(
      Line = 'Content',
      Code = 'Content',
    )
    
    def initialize(self, path, *components, **options):
        self.path = path
        map(self.add, components)
        self._provides = options.pop('provides', path)
        if options: self.warning('%s unused options: %s\n' % (self.__class__.__name__, options))
        return self

    def finalize(self):
        self._provides = self.get_path() or self._provides

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,[self.path]+[c for (c,l) in self.components])))

def _test():
    import doctest
    doctest.testmod()
    
if __name__ == "__main__":
    _test()


"""Support for parametric tests in unittest.

:Author: Fernando Perez

Purpose
=======

Briefly, the main class in this module allows you to easily and cleanly
(without the gross name-mangling hacks that are normally needed) to write
unittest TestCase classes that have parametrized tests.  That is, tests which
consist of multiple sub-tests that scan for example a parameter range, but
where you want each sub-test to:

* count as a separate test in the statistics.

* be run even if others in the group error out or fail.


The class offers a simple name-based convention to create such tests (see
simple example at the end), in one of two ways:

* Each sub-test in a group can be run fully independently, with the
  setUp/tearDown methods being called each time.

* The whole group can be run with setUp/tearDown being called only once for the
  group.  This lets you conveniently reuse state that may be very expensive to
  compute for multiple tests.  Be careful not to corrupt it!!!


Caveats
=======

This code relies on implementation details of the unittest module (some key
methods are heavily modified versions of those, after copying them in).  So it
may well break either if you make sophisticated use of the unittest APIs, or if
unittest itself changes in the future.  I have only tested this with Python
2.5.

"""
__docformat__ = "restructuredtext en"

