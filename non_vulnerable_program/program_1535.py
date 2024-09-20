from statements import *
from typedecl_statements import *

access_spec = [Public, Private]

interface_specification = [Function, Subroutine,
                           ModuleProcedure
                           ]

module_subprogram_part = [
    Contains,
    Function,
    Subroutine
    ]

specification_stmt = [
    # Access, Allocatable, Asynchronous, Bind, Common,
    Data, Dimension,
    Equivalence, #External, Intent
    # Intrinsic, Namelist, Optional, Pointer, Protected,
    Save, #Target, Volatile, Value
    ]
intrinsic_type_spec = [
    Integer , Real, DoublePrecision, Complex, Character, Logical
    ]
declaration_type_spec = intrinsic_type_spec + [
    TypeStmt,
    Class
    ]
type_declaration_stmt = declaration_type_spec

private_or_sequence = [
    Private, #Sequence
    ]

component_part = declaration_type_spec + [
    #Procedure
    ]

type_bound_procedure_part = [
    Contains, Private, #Procedure, Generic, Final
    ]

#R214
action_stmt = [
    Allocate,
    Assignment, #PointerAssignment,
    Backspace,
    Call,
    Close,
    Continue,
    Cycle,
    Deallocate,
    Endfile, #EndFunction, EndProgram, EndSubroutine,
    Exit,
    # Flush, Forall,
    Goto, If, #Inquire,
    Nullify,
    Open, 
    Print, Read,
    Return,
    Rewind,
    Stop, #Wait, Where,
    Write,
    # arithmetic-if-stmt, computed-goto-stmt
    ]

executable_construct = action_stmt + [
    # Associate, Case,
    Do,
    # Forall,
    IfThen,
    Select, #Where
    ]
execution_part_construct = executable_construct + [
    Format, #Entry, Data
    ]
execution_part = execution_part_construct[:]

#C201, R208
for cls in [EndFunction, EndProgram, EndSubroutine]:
    try: execution_part.remove(cls)
    except ValueError: pass

internal_subprogram = [Function, Subroutine]
internal_subprogram_part = [
    Contains,
    ] + internal_subprogram

declaration_construct = [
    TypeDecl, #Entry, Enum,
    Format,
    Interface,
    Parameter, #Procedure,
    ] + specification_stmt + type_declaration_stmt # stmt-function-stmt
implicit_part = [
    Implicit, Parameter, Format, #Entry
    ]
specification_part = [
    Use, #Import 
    ] + implicit_part + declaration_construct
external_subprogram = [Function, Subroutine]
main_program = [Program] + specification_part + execution_part + internal_subprogram_part
program_unit = main_program + external_subprogram + [Module,
                                                     #BlockData
                                                     ]



