       from NumPy version ...
       ExtGen is developed by Pearu Peterson <pearu.peterson@gmail.com>.
       For more information see http://www.scipy.org/ExtGen/ .
    */
    #ifdef __cplusplus
    extern "C" {
    #endif
    #ifdef __cplusplus
    }
    #endif
    <BLANKLINE>
    """

    container_options = dict(
        CHeader = dict(default='<KILLLINE>', prefix='\n/* CHeader */\n', skip_prefix_when_empty=True),
        CTypeDef = dict(default='<KILLLINE>', prefix='\n/* CTypeDef */\n', skip_prefix_when_empty=True),
        CProto = dict(default='<KILLLINE>', prefix='\n/* CProto */\n', skip_prefix_when_empty=True),
        CDefinition = dict(default='<KILLLINE>', prefix='\n/* CDefinition */\n', skip_prefix_when_empty=True),
        CDeclaration = dict(default='<KILLLINE>', separator=';\n', suffix=';',
                            prefix='\n/* CDeclaration */\n', skip_prefix_when_empty=True),
        CMainProgram = dict(default='<KILLLINE>', prefix='\n/* CMainProgram */\n', skip_prefix_when_empty=True),
        )

    template_c_header = '''\
/* -*- c -*- */
/* This file %(path)r is generated using ExtGen tool
