from statements import *
from typedecl_statements import *

f2py_stmt = [Threadsafe, FortranName, Depend, Check, CallStatement,
             CallProtoArgument]

access_spec = [Public, Private]

interface_specification = [Function, Subroutine,
                           ModuleProcedure
                           ]

module_subprogram_part = [ Contains, Function, Subroutine ]

specification_stmt = access_spec + [ Allocatable, Asynchronous, Bind,
    Common, Data, Dimension, Equivalence, External, Intent, Intrinsic,
    Namelist, Optional, Pointer, Protected, Save, Target, Volatile,
    Value ]

intrinsic_type_spec = [ SubprogramPrefix, Integer , Real,
    DoublePrecision, Complex, DoubleComplex, Character, Logical, Byte
    ]

derived_type_spec = [  ]
type_spec = intrinsic_type_spec + derived_type_spec
declaration_type_spec = intrinsic_type_spec + [ TypeStmt, Class ]
    
type_declaration_stmt = declaration_type_spec

private_or_sequence = [ Private, Sequence ]

component_part = declaration_type_spec + [ ModuleProcedure ]

proc_binding_stmt = [SpecificBinding, GenericBinding, FinalBinding]

type_bound_procedure_part = [Contains, Private] + proc_binding_stmt

#R214
action_stmt = [ Allocate, GeneralAssignment, Assign, Backspace, Call, Close,
    Continue, Cycle, Deallocate, Endfile, Exit, Flush, ForallStmt,
    Goto, If, Inquire, Nullify, Open, Print, Read, Return, Rewind,
    Stop, Wait, WhereStmt, Write, ArithmeticIf, ComputedGoto,
    AssignedGoto, Pause ]
# GeneralAssignment = Assignment + PointerAssignment
# EndFunction, EndProgram, EndSubroutine - part of the corresponding blocks

executable_construct = [ Associate, Do, ForallConstruct, IfThen,
    Select, WhereConstruct ] + action_stmt
#Case, see Select

execution_part_construct = executable_construct + [ Format, Entry,
    Data ]

execution_part = execution_part_construct[:]

#C201, R208
for cls in [EndFunction, EndProgram, EndSubroutine]:
    try: execution_part.remove(cls)
    except ValueError: pass

internal_subprogram = [Function, Subroutine]

internal_subprogram_part = [ Contains, ] + internal_subprogram

declaration_construct = [ TypeDecl, Entry, Enum, Format, Interface,
    Parameter, ModuleProcedure, ] + specification_stmt + \
    type_declaration_stmt
# stmt-function-stmt

implicit_part = [ Implicit, Parameter, Format, Entry ]

specification_part = [ Use, Import ] + implicit_part + \
                     declaration_construct


external_subprogram = [Function, Subroutine]

main_program = [Program] + specification_part + execution_part + \
               internal_subprogram_part

program_unit = main_program + external_subprogram + [Module,
                                                     BlockData ]


#!/usr/bin/env python
"""
Defines FortranReader classes for reading Fortran codes from
files and strings. FortranReader handles comments and line continuations
of both fix and free format Fortran codes.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: May 2006
-----
"""

__all__ = ['FortranFileReader',
           'FortranStringReader',
           'FortranReaderError',
           'Line', 'SyntaxErrorLine',
           'Comment',
           'MultiLine','SyntaxErrorMultiLine',
           ]

