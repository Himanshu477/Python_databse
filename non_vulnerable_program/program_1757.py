                from/to Fortran reader.
      isvalid - boolean, when False, the Block instance will be ignored.

      stmt_cls, end_stmt_cls

    """
    _repr_attr_names = ['blocktype','name'] + Statement._repr_attr_names
    def __init__(self, parent, item=None):

        self.content = []
        self.get_item = parent.get_item # get line function
        self.put_item = parent.put_item # put line function
        if not hasattr(self, 'blocktype'):
            self.blocktype = self.__class__.__name__.lower()
        if not hasattr(self, 'name'):
            # process_item may change this
            self.name = '__'+self.blocktype.upper()+'__' 
        Statement.__init__(self, parent, item)
        return

    def tostr(self):
        return self.blocktype.upper() + ' '+ self.name
    
    def __str__(self):
        l=[self.get_indent_tab(colon=':') + self.tostr()]
        for c in self.content:
            l.append(str(c))
        return '\n'.join(l)

    def torepr(self, depth=-1, incrtab=''):
        tab = incrtab + self.get_indent_tab()
        ttab = tab + '  '
        l=[Statement.torepr(self, depth=depth,incrtab=incrtab)]
        if depth==0 or not self.content:
            return '\n'.join(l)
        l.append(ttab+'content:')
        for c in self.content:
            if isinstance(c,EndStatement):
                l.append(c.torepr(depth-1,incrtab))
            else:
                l.append(c.torepr(depth-1,incrtab + '  '))
        return '\n'.join(l)

    def process_item(self):
        """ Process the line
        """
        item = self.item
        if item is None: return
        self.fill()
        return

    def fill(self, end_flag = False):
        """
        Fills blocks content until the end of block statement.
        """

        mode = self.reader.mode
        classes = self.get_classes()
        self.classes = [cls for cls in classes if mode in cls.modes]
        self.pyf_classes = [cls for cls in classes if 'pyf' in cls.modes]

        item = self.get_item()
        while item is not None:
            if isinstance(item, Line):
                if self.process_subitem(item):
                    end_flag = True
                    break
            item = self.get_item()

        if not end_flag:
            self.warning('failed to find the end of block')
        return

    def process_subitem(self, item):
        """
        Check is item is blocks start statement, if it is, read the block.

        Return True to stop adding items to given block.
        """
        line = item.get_line()

        # First check for the end of block
        cls = self.end_stmt_cls
        if cls.match(line):
            stmt = cls(self, item)
            if stmt.isvalid:
                self.content.append(stmt)
                return True

        if item.is_f2py_directive:
            classes = self.pyf_classes
        else:
            classes = self.classes

        # Look for statement match
        for cls in classes:
            if cls.match(line):
                stmt = cls(self, item)
                if stmt.isvalid:
                    if not stmt.ignore:
                        self.content.append(stmt)
                    return False
                # item may be cloned that changes the items line:
                line = item.get_line()
                
        # Check if f77 code contains inline comments or other f90
        # constructs that got undetected by get_source_info.
        if item.reader.isfix77:
            i = line.find('!')
            if i != -1:
                message = item.reader.format_message(\
                        'WARNING',
                        'no parse pattern found for "%s" in %r block'\
                        ' maybe due to inline comment.'\
                        ' Trying to remove the comment.'\
                        % (item.get_line(),self.__class__.__name__),
                        item.span[0], item.span[1])
                # .. but at the expense of loosing the comment.
                self.show_message(message)
                newitem = item.copy(line[:i].rstrip())
                return self.process_subitem(newitem)

            # try fix90 statement classes
            f77_classes = self.classes
            classes = []
            for cls in self.get_classes():
                if 'fix90' in cls.modes and cls not in f77_classes:
                    classes.append(cls)
            if classes:
                message = item.reader.format_message(\
                        'WARNING',
                        'no parse pattern found for "%s" in %r block'\
                        ' maybe due to strict f77 mode.'\
                        ' Trying f90 fix mode patterns..'\
                        % (item.get_line(),self.__class__.__name__),
                        item.span[0], item.span[1])
                self.show_message(message)
    
                item.reader.set_mode(False, False)
                self.classes = classes
            
                r = BeginStatement.process_subitem(self, item)
                if r is None:
                    # restore f77 fix mode
                    self.classes = f77_classes
                    item.reader.set_mode(False, True)
                else:
                    message = item.reader.format_message(\
                        'INFORMATION',
                        'The f90 fix mode resolved the parse pattern issue.'\
                        ' Setting reader to f90 fix mode.',
                        item.span[0], item.span[1])
                    self.show_message(message)
                    # set f90 fix mode
                    self.classes = f77_classes + classes
                    self.reader.set_mode(False, False)
                return r

        self.handle_unknown_item(item)
        return

    def handle_unknown_item(self, item):
        message = item.reader.format_message(\
                        'WARNING',
                        'no parse pattern found for "%s" in %r block.'\
                        % (item.get_line(),self.__class__.__name__),
                        item.span[0], item.span[1])
        self.show_message(message)
        self.content.append(item)
        #sys.exit()
        return

    def analyze(self):
        for stmt in self.content:
            stmt.analyze()
        return

class EndStatement(Statement):
    """
    END [<blocktype> [<name>]]

    EndStatement instances have additional attributes:
      name
      blocktype
    """
    _repr_attr_names = ['blocktype','name'] + Statement._repr_attr_names

    def __init__(self, parent, item):
        if not hasattr(self, 'blocktype'):
            self.blocktype = self.__class__.__name__.lower()[3:]
        Statement.__init__(self, parent, item)

    def process_item(self):
        item = self.item
        line = item.get_line().replace(' ','')[3:]
        blocktype = self.blocktype
        if line.startswith(blocktype):
            line = line[len(blocktype):].strip()
        else:
            if line:
                # not the end of expected block
                line = ''
                self.isvalid = False
        if line:
            if not line==self.parent.name:
                self.warning(\
                    'expected the end of %r block but got the end of %r, skipping.'\
                    % (self.parent.name, line))
                self.isvalid = False
        self.name = self.parent.name

    def analyze(self):
        return

    def get_indent_tab(self,colon=None,deindent=False):
        return Statement.get_indent_tab(self, colon=colon, deindent=True)

    def __str__(self):
        return self.get_indent_tab() + 'END %s %s'\
               % (self.blocktype.upper(),self.name or '')



"""
Fortran block statements.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: May 2006
-----
"""

__all__ = ['BeginSource','Module','PythonModule','Program','BlockData','Interface',
           'Subroutine','Function','Select','EndWhere','WhereConstruct','ForallConstruct',
           'IfThen','If','Do','Associate','TypeDecl','Enum']

