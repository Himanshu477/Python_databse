from numpy import newaxis

NewAxis = newaxis



# missing Numarray defined names (in from numarray import *)
##__all__ = ['ArrayType', 'CLIP', 'ClassicUnpickler', 'Complex32_fromtype',
##           'Complex64_fromtype', 'ComplexArray', 'EarlyEOFError', 'Error',
##           'FileSeekWarning', 'MAX_ALIGN', 'MAX_INT_SIZE', 'MAX_LINE_WIDTH',
##           'MathDomainError', 'NDArray', 'NewArray', 'NewAxis', 'NumArray',
##           'NumError', 'NumOverflowError', 'PRECISION', 'Py2NumType',
##           'PyINT_TYPES', 'PyLevel2Type', 'PyNUMERIC_TYPES', 'PyREAL_TYPES',
##           'RAISE', 'SLOPPY', 'STRICT', 'SUPPRESS_SMALL', 'SizeMismatchError',
##           'SizeMismatchWarning', 'SuitableBuffer', 'USING_BLAS',
##           'UnderflowError', 'UsesOpPriority', 'WARN', 'WRAP', 'all',
##           'allclose', 'alltrue', 'and_', 'any', 'arange', 'argmax',
##           'argmin', 'argsort', 'around', 'array2list', 'array_equal',
##           'array_equiv', 'array_repr', 'array_str', 'arrayprint',
##           'arrayrange', 'average', 'choose', 'clip',
##           'codegenerator', 'compress', 'concatenate', 'conjugate',
##           'copy', 'copy_reg', 'diagonal', 'divide_remainder',
##           'dotblas', 'e', 'explicit_type', 'flush_caches', 'fromfile',
##           'fromfunction', 'fromlist', 'fromstring', 'generic',
##           'genericCoercions', 'genericPromotionExclusions', 'genericTypeRank',
##           'getShape', 'getTypeObject', 'handleError', 'identity', 'indices',
##           'info', 'innerproduct', 'inputarray', 'isBigEndian',
##           'kroneckerproduct', 'lexsort', 'libnumarray', 'libnumeric',
##           'load', 'make_ufuncs', 'math', 'memory',
##           'numarrayall', 'numarraycore', 'numerictypes', 'numinclude',
##           'operator', 'os', 'outerproduct', 'pi', 'put', 'putmask',
##           'pythonTypeMap', 'pythonTypeRank', 'rank', 'repeat',
##           'reshape', 'resize', 'round', 'safethread', 'save', 'scalarTypeMap',
##           'scalarTypes', 'searchsorted', 'session', 'shape', 'sign', 'size',
##           'sometrue', 'sort', 'swapaxes', 'sys', 'take', 'tcode',
##           'tensormultiply', 'tname', 'trace', 'transpose', 'typeDict',
##           'typecode', 'typecodes', 'typeconv', 'types', 'ufunc',
##           'ufuncFactory', 'value', ]


__all__ = ['asarray', 'ones', 'zeros', 'array', 'where']
__all__ += ['vdot', 'dot', 'matrixmultiply', 'ravel', 'indices',
            'arange', 'concatenate']

