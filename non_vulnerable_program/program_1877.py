from base import Base

class CCode(Base):

    """
    CCode(*lines, provides=..)
    """

    container_options = dict(CCodeLines=dict())

    template = '%(CCodeLines)s'

    def initialize(self, *lines, **options):
        self.lines = []
        map(self.add, lines)

    def update_containers(self):
        CCodeLines = self.get_container('CCodeLines')
        CCodeLines.add('\n'.join(self.lines))

    def add(self, component, label=None):
        if isinstance(component, str):
            assert label is None,`label`
            self.lines.append(component)
        elif isinstance(component, CCode):
            assert label is None,`label`
            self.lines.extend(component.lines)
        else:
            Base.add(self, component. label)

        


"""
Defines C type declaration templates:

  CTypeAlias(name, ctype)  --- typedef ctype name;
  CTypeFunction(name, rtype, atype1, atype2,..) --- typedef rtype (*name)(atype1, atype2,...);
  CTypeStruct(name, (name1,type1), (name2,type2), ...) --- typedef struct { type1 name1; type2 name2; .. } name;
  CTypePtr(ctype) --- ctype *
  CInt(), CLong(), ... --- int, long, ...
  CPyObject()

The instances of CTypeBase have the following public methods and properties:

  - .asPtr()
  - .declare(name)
"""


