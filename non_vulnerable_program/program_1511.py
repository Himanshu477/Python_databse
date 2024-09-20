      import_list  - list of names
    """
    start_re = re.compile(r'import\b').match
    def __init__(self, parent, item):
        Statement.__init__(self, parent, item)
        line = item.get_line()[6:].lstrip()
        if line.startswith('::'): line = line[2:]
        self.import_list = []
        for n in line.split(','):
            n = n.strip()
            if not n: continue
            self.import_list.append(n)

    def __str__(self):
        s = self.get_intent_tab() + 'IMPORT'
        if self.import_list:
            s += ' :: ' + ','.join(self.import_list)
        return s

class Assignment(Statement):
    """Assignment statement class

    <variable> = <expr>
    <pointer variable> => <expr>
    
    Assignment instance has attributes:
      variable
      sign
      expr
    """
    start_re = re.compile(r'\w[\w%]*(\s*\(\s*[^)]*\)|)\s*=\>?',re.I).match
    stmt_re = re.compile(r'(?P<variable>\w[\w%]*(\s*\(\s*[^)]*\)|))\s*(?P<sign>=\>?)\s*(?P<expr>.*)\Z',re.I).match
    def __init__(self, parent, item):
        Statement.__init__(self, parent, item)
        line = item.get_line()
        m = self.stmt_re(line)
        self.variable = m.group('variable')
        self.sign = m.group('sign')
        self.expr = m.group('expr')
    def __str__(self):
        return self.get_intent_tab() + '%s %s %s' \
               % (self.variable, self.sign, self.expr)

class If(Statement):
    """If statement class
    IF (<scalar-logical-expr>) <action-stmt>

    If instance has attributes:
      expr
      stmt
    """
    start_re = re.compile(r'if\b',re.I).match
    def __init__(self, parent, item):
        Statement.__init__(self, parent, item)
        line = item.get_line()[2:].strip()
        i = line.find(')')
        self.expr = line[1:i].strip()
        self.stmt = line[i+1:].strip()
    def __str__(self):
        return self.get_intent_tab()+ 'IF (%s) %s' % (self.expr, self.stmt)

class Call(Statement):
    """Call statement class
    CALL <proc-designator> [([arg-spec-list])]

    Call instance has attributes:
      designator
      arg_list
    """
    start_re = re.compile(r'call\b', re.I).match
    def __init__(self, parent, item):
        Statement.__init__(self, parent, item)
        line = item.get_line()[4:].strip()
        i = line.find('(')
        self.arg_list = []
        if i==-1:
            self.designator = line.strip()
        else:
            self.designator = line[:i].strip()
            for n in line[i+1:-1].split(','):
                n = n.strip()
                if not n: continue
                self.arg_list.append(n)
    def __str__(self):
        s = self.get_intent_tab() + 'CALL '+str(self.designator)
        if self.arg_list:
            s += '('+', '.join(map(str,self.arg_list))+ ')'
        return s

class Contains(Statement):
    """
    CONTAINS
    """
    start_re = re.compile(r'contains\Z',re.I).match
    
    def __str__(self): return self.get_intent_tab() + 'CONTAINS'

class Continue(Statement):
    """
    CONTINUE
    """
    start_re = re.compile(r'continue\Z',re.I).match
    
    def __str__(self): return self.get_intent_tab() + 'CONTINUE'

class Return(Statement):
    """
    RETURN [<int-expr>]

    Return instance has attributes:
      expr
    """
    start_re = re.compile(r'return\b',re.I).match
    
    def __init__(self, parent, item):
        Statement.__init__(self, parent, item)
        line = item.get_line()[6:].strip()
        self.expr = ''
        if line:
            self.expr = line
    def __str__(self):
        s = self.get_intent_tab() + 'RETURN'
        if self.expr:
            s += ' ' + str(self.expr)
        return s

class Stop(Statement):
    """
    STOP [<stop-code>]

    Return instance has attributes:
      code
    """
    start_re = re.compile(r'stop\b',re.I).match
    def __init__(self, parent, item):
        Statement.__init__(self, parent, item)
        line = item.get_line()[4:].strip()
        self.code = ''
        if line:
            self.code = line
    def __str__(self):
        s = self.get_intent_tab() + 'STOP'
        if self.code:
            s += ' ' + str(self.code)
        return s

class Format(Statement):
    """
    FORMAT ( [<format-item-list>] )

    Return instance has attributes:
      spec
    """
    start_re = re.compile(r'format\s*\([^)]*\)\Z',re.I).match
    def __init__(self, parent, item):
        Statement.__init__(self, parent, item)
        line = item.get_line()[6:].strip()
        assert line[0]=='(' and line[-1]==')',`line`
        self.spec = line[1:-1].strip()
    def __str__(self):
        return self.get_intent_tab() + 'FORMAT (%s)' % (self.spec)

end_stmts = {'ModuleBlock':EndModule, 'PythonModuleBlock':EndPythonModule,
             'TypeBlock':EndType,
             'SubroutineBlock':EndSubroutine,'FunctionBlock':EndFunction,
             'IfThenBlock':EndIfThen,'DoBlock':EndDo,'InterfaceBlock':EndInterface,
             'SelectBlock':EndSelect}

block_stmts = {'IfThenBlock':IfThen}

exec_stmts = [Assignment, If, Call, Stop, Continue]
statements = {}
statements['Block'] = []
statements['ProgramBlock'] = [Use, Contains] + exec_stmts
# Format, misc_decl, Data, derived_type_defs, Interface_block, exec_stmt, Contains, StmtFunc_stmt
statements['ModuleBlock'] = [Use, Contains]
statements['BlockDataBlock'] = [Use]
statements['SubroutineBlock'] = [Use, Contains, Return, Format] + exec_stmts
statements['FunctionBlock'] = statements['SubroutineBlock']
statements['InterfaceBlock'] = [Use, Import]
statements['PythonModuleBlock'] = [Use]
statements['TypeBlock'] = [Contains]
for n in ['IfThenBlock','DoBlock','SelectBlock']:
    statements[n] = exec_stmts



__all__ = ['fft', 'fft2d', 'fftnd', 'hermite_fft', 'inverse_fft', 'inverse_fft2d',
           'inverse_fftnd', 'inverse_hermite_fft', 'inverse_real_fft', 'inverse_real_fft2d',
           'inverse_real_fftnd', 'real_fft', 'real_fft2d', 'real_fftnd']

