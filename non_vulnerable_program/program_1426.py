import __version__
f2py_version = __version__.version

"""
 Usage of crackfortran:
 ======================
 Command line keys: -quiet,-verbose,-fix,-f77,-f90,-show,-h <pyffilename>
                    -m <module name for f77 routines>,--ignore-contains
 Functions: crackfortran, crack2fortran
 The following Fortran statements/constructions are supported
 (or will be if needed):
    block data,byte,call,character,common,complex,contains,data,
    dimension,double complex,double precision,end,external,function,
    implicit,integer,intent,interface,intrinsic,
    logical,module,optional,parameter,private,public,
    program,real,(sequence?),subroutine,type,use,virtual,
    include,pythonmodule
 Note: 'virtual' is mapped to 'dimension'.
 Note: 'implicit integer (z) static (z)' is 'implicit static (z)' (this is minor bug).
 Note: code after 'contains' will be ignored until its scope ends.
 Note: 'common' statement is extended: dimensions are moved to variable definitions
 Note: f2py directive: <commentchar>f2py<line> is read as <line>
 Note: pythonmodule is introduced to represent Python module

 Usage:
   `postlist=crackfortran(files,funcs)`
   `postlist` contains declaration information read from the list of files `files`.
   `crack2fortran(postlist)` returns a fortran code to be saved to pyf-file

   `postlist` has the following structure:
  *** it is a list of dictionaries containing `blocks':
      B = {'block','body','vars','parent_block'[,'name','prefix','args','result',
           'implicit','externals','interfaced','common','sortvars',
           'commonvars','note']}
      B['block'] = 'interface' | 'function' | 'subroutine' | 'module' |
                   'program' | 'block data' | 'type' | 'pythonmodule'
      B['body'] --- list containing `subblocks' with the same structure as `blocks'
      B['parent_block'] --- dictionary of a parent block:
                              C['body'][<index>]['parent_block'] is C
      B['vars'] --- dictionary of variable definitions
      B['sortvars'] --- dictionary of variable definitions sorted by dependence (independent first)
      B['name'] --- name of the block (not if B['block']=='interface')
      B['prefix'] --- prefix string (only if B['block']=='function')
      B['args'] --- list of argument names if B['block']== 'function' | 'subroutine'
      B['result'] --- name of the return value (only if B['block']=='function')
      B['implicit'] --- dictionary {'a':<variable definition>,'b':...} | None
      B['externals'] --- list of variables being external
      B['interfaced'] --- list of variables being external and defined
      B['common'] --- dictionary of common blocks (list of objects)
      B['commonvars'] --- list of variables used in common blocks (dimensions are moved to variable definitions)
      B['from'] --- string showing the 'parents' of the current block
      B['use'] --- dictionary of modules used in current block:
          {<modulename>:{['only':<0|1>],['map':{<local_name1>:<use_name1>,...}]}}
      B['note'] --- list of LaTeX comments on the block
      B['f2pyenhancements'] --- optional dictionary
           {'threadsafe':'','fortranname':<name>,
            'callstatement':<C-expr>|<multi-line block>,
            'callprotoargument':<C-expr-list>,
            'usercode':<multi-line block>|<list of multi-line blocks>,
            'pymethoddef:<multi-line block>'
            }
      B['entry'] --- dictionary {entryname:argslist,..}
      B['varnames'] --- list of variable names given in the order of reading the
                        Fortran code, useful for derived types.
  *** Variable definition is a dictionary
      D = B['vars'][<variable name>] =
      {'typespec'[,'attrspec','kindselector','charselector','=','typename']}
      D['typespec'] = 'byte' | 'character' | 'complex' | 'double complex' |
                      'double precision' | 'integer' | 'logical' | 'real' | 'type'
      D['attrspec'] --- list of attributes (e.g. 'dimension(<arrayspec>)',
                        'external','intent(in|out|inout|hide|c|callback|cache)',
                        'optional','required', etc)
      K = D['kindselector'] = {['*','kind']} (only if D['typespec'] =
                          'complex' | 'integer' | 'logical' | 'real' )
      C = D['charselector'] = {['*','len','kind']}
                              (only if D['typespec']=='character')
      D['='] --- initialization expression string
      D['typename'] --- name of the type if D['typespec']=='type'
      D['dimension'] --- list of dimension bounds
      D['intent'] --- list of intent specifications
      D['depend'] --- list of variable names on which current variable depends on
      D['check'] --- list of C-expressions; if C-expr returns zero, exception is raised
      D['note'] --- list of LaTeX comments on the variable
  *** Meaning of kind/char selectors (few examples):
      D['typespec>']*K['*']
      D['typespec'](kind=K['kind'])
      character*C['*']
      character(len=C['len'],kind=C['kind'])
      (see also fortran type declaration statement formats below)

 Fortran 90 type declaration statement format (F77 is subset of F90)
====================================================================
 (Main source: IBM XL Fortran 5.1 Language Reference Manual)
 type declaration = <typespec> [[<attrspec>]::] <entitydecl>
 <typespec> = byte                          |
              character[<charselector>]     |
              complex[<kindselector>]       |
              double complex                |
              double precision              |
              integer[<kindselector>]       |
              logical[<kindselector>]       |
              real[<kindselector>]          |
              type(<typename>)
 <charselector> = * <charlen>               |
              ([len=]<len>[,[kind=]<kind>]) |
              (kind=<kind>[,len=<len>])
 <kindselector> = * <intlen>                |
              ([kind=]<kind>)
 <attrspec> = comma separated list of attributes.
              Only the following attributes are used in
              building up the interface:
                 external
                 (parameter --- affects '=' key)
                 optional
                 intent
              Other attributes are ignored.
 <intentspec> = in | out | inout
 <arrayspec> = comma separated list of dimension bounds.
 <entitydecl> = <name> [[*<charlen>][(<arrayspec>)] | [(<arrayspec>)]*<charlen>]
                       [/<init_expr>/ | =<init_expr>] [,<entitydecl>]

 In addition, the following attributes are used: check,depend,note

 TODO:
     * Apply 'parameter' attribute (e.g. 'integer parameter :: i=2' 'real x(i)'
                                    -> 'real x(2)')
     The above may be solved by creating appropriate preprocessor program, for example.
"""
#
