import re
import sys
from readfortran import Line


class Statement:
    """
    Statement instance has attributes:
      parent  - Parent BeginStatement or FortranParser instance
      item    - Line instance containing the statement line
      isvalid - boolean, when False, the Statement instance will be ignored
    """
    modes = ['free90','fix90','fix77','pyf']

    def __init__(self, parent, item):
        self.parent = parent
        self.reader = parent.reader
        self.item = item

        # If statement instance is constructed by error, set isvalid to False
        self.isvalid = True

        self.process_item()

    def get_indent_tab(self,colon='',deindent=False):
        if self.reader.isfix:
            tab = ' '*6
        else:
            tab = ''
        p = self.parent
        while isinstance(p, Statement):
            tab += '  '
            p = p.parent
        if deindent:
            tab = tab[:-2]
        if self.item is None:
            return tab
        s = self.item.label
        if s:
            c = ''
            if self.reader.isfix:
                c = ' '
            tab = tab[len(c+s)+len(colon):]
            if not tab: tab = ' '
            tab = c + s + colon + tab
        return tab

class BeginStatement(Statement):
    """ <blocktype> <name>

    BeginStatement instances have additional attributes:
      name
      blocktype

    Block instance has attributes:
      content - list of Line or Statement instances
      name    - name of the block, unnamed blocks are named
                with the line label
      parent  - Block or FortranParser instance
      item    - Line instance containing the block start statement
      get_item, put_item - methods to retrive/submit Line instaces
