import math
functiondefs = {
    (len,(String,)):
    Function_Descriptor(code='strlen(%s)',return_type=Integer),
    
    (len,(LongVector,)):
    Function_Descriptor(code='PyArray_Size((PyObject*)%s)',return_type=Integer),

    (float,(Integer,)):
    Function_Descriptor(code='(double)(%s)',return_type=Double),
    
    (range,(Integer,Integer)):
    Function_Descriptor(code='XRange(%s)',return_type=XRange),

    (range,(Integer)):
    Function_Descriptor(code='XRange(%s)',return_type=XRange),

    (math.sin,(Double,)):
    Function_Descriptor(code='sin(%s)',return_type=Double),

    (math.cos,(Double,)):
    Function_Descriptor(code='cos(%s)',return_type=Double),

    (math.sqrt,(Double,)):
    Function_Descriptor(code='sqrt(%s)',return_type=Double),
    }
    


##################################################################
#                      FUNCTION LOOKUP_TYPE                      #
##################################################################
def lookup_type(x):
    T = type(x)
    try:
        return typedefs[T]
    except:
        import scipy.base as nx
        if isinstance(T,nx.ArrayType):
            return typedefs[(T,len(x.shape),x.dtypechar)]
        elif T == InstanceType:
            return Instance(x)
        else:
            raise NotImplementedError,T

##################################################################
#                        class ACCELERATE                        #
##################################################################
class accelerate:
    
    def __init__(self, function, *args, **kw):
        assert type(function) == FunctionType
        self.function = function
        self.module = inspect.getmodule(function)
        if self.module is None:
            import __main__
            self.module = __main__
        self.__call_map = {}

    def __cache(self,*args):
        raise TypeError

    def __call__(self,*args):
        try:
            return self.__cache(*args)
        except TypeError:
            # Figure out type info -- Do as tuple so its hashable
            signature = tuple( map(lookup_type,args) )
            
            # If we know the function, call it
            try:
                fast = self.__call_map[signature]
            except:
                fast = self.singleton(signature)
                self.__cache = fast
                self.__call_map[signature] = fast
            return fast(*args)

    def signature(self,*args):
        # Figure out type info -- Do as tuple so its hashable
        signature = tuple( map(lookup_type,args) )
        return self.singleton(signature)


    def singleton(self,signature):
        identifier = self.identifier(signature)
        
        # Generate a new function, then call it
        f = self.function

        # See if we have an accelerated version of module
        try:
            print 'lookup',self.module.__name__+'_weave'
            accelerated_module = __import__(self.module.__name__+'_weave')
            print 'have accelerated',self.module.__name__+'_weave'
            fast = getattr(accelerated_module,identifier)
            return fast
        except ImportError:
            accelerated_module = None
        except AttributeError:
            pass

        P = self.accelerate(signature,identifier)

        E = weave.ext_tools.ext_module(self.module.__name__+'_weave')
        E.add_function(P)
        E.generate_file()
        weave.build_tools.build_extension(self.module.__name__+'_weave.cpp',verbose=2)

        if accelerated_module:
            raise NotImplementedError,'Reload'
        else:
            accelerated_module = __import__(self.module.__name__+'_weave')

        fast = getattr(accelerated_module,identifier)
        return fast

    def identifier(self,signature):
        # Build an MD5 checksum
        f = self.function
        co = f.func_code
        identifier = str(signature)+\
                     str(co.co_argcount)+\
                     str(co.co_consts)+\
                     str(co.co_varnames)+\
                     co.co_code
        return 'F'+md5.md5(identifier).hexdigest()
        
    def accelerate(self,signature,identifier):
        P = Python2CXX(self.function,signature,name=identifier)
        return P

    def code(self,*args):
        if len(args) != self.function.func_code.co_argcount:
            raise TypeError,'%s() takes exactly %d arguments (%d given)'%(
                self.function.__name__,
                self.function.func_code.co_argcount,
                len(args))
        signature = tuple( map(lookup_type,args) )
        ident = self.function.__name__
        return self.accelerate(signature,ident).function_code()
        

##################################################################
#                        CLASS PYTHON2CXX                        #
##################################################################
class Python2CXX(CXXCoder):
    def typedef_by_value(self,v):
        T = lookup_type(v)
        if T not in self.used:
            self.used.append(T)
        return T

    def function_by_signature(self,signature):
        descriptor = functiondefs[signature]
        if descriptor.return_type not in self.used:
            self.used.append(descriptor.return_type)
        return descriptor

    def __init__(self,f,signature,name=None):
        # Make sure function is a function
        import types
        assert type(f) == FunctionType
        # and check the input type signature
        assert reduce(lambda x,y: x and y,
                      map(lambda x: isinstance(x,Type_Descriptor),
                          signature),
                      1),'%s not all type objects'%signature
        self.arg_specs = []
        self.customize = weave.base_info.custom_info()

        CXXCoder.__init__(self,f,signature,name)

        return

    def function_code(self):
        code = self.wrapped_code()
        for T in self.used:
            if T != None and T.module_init_code:
                self.customize.add_module_init_code(T.module_init_code)
        return code

    def python_function_definition_code(self):
        return '{ "%s", wrapper_%s, METH_VARARGS, %s },\n'%(
            self.name,
            self.name,
            CStr(self.function.__doc__))


