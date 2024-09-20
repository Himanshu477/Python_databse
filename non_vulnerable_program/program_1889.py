   from NumPy version %(numpy_version)s.
   ExtGen is developed by Pearu Peterson <pearu.peterson@gmail.com>.
   For more information see http://www.scipy.org/ExtGen/ .
*/'''


    template = template_c_header + '''
#ifdef __cplusplus
extern \"C\" {
#endif
%(CHeader)s
%(CTypeDef)s
%(CProto)s
%(CDefinition)s
%(CDeclaration)s
%(CMainProgram)s
#ifdef __cplusplus
}
#endif
'''

    component_container_map = dict(
      CHeader = 'CHeader',
      CFunction = 'CDefinition',
      CDeclaration = 'CDeclaration',
    )




def _test():
    import doctest
    doctest.testmod()
    
if __name__ == "__main__":
    _test()



__all__ = ['PySource', 'PyCFunction', 'PyCModule', 'PyCTypeSpec', 'PyCArgument', 'PyCReturn']

