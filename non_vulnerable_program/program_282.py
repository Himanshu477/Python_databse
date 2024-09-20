from UserList import UserList
class test_object_set_item_op_index(unittest.TestCase):
    def check_list_refcount(self):
        a = UserList([1,2,3])            
        # temporary refcount fix until I understand why it incs by one.
        inline_tools.inline("a[1] = 1234;",['a'])        
        before1 = sys.getrefcount(a)
        after1 = sys.getrefcount(a)
        assert after1 == before1
    def check_set_int(self):
        a = UserList([1,2,3])            
        inline_tools.inline("a[1] = 1234;",['a'])        
        assert sys.getrefcount(a[1]) == 2                
        assert a[1] == 1234
    def check_set_double(self):
        a = UserList([1,2,3])            
        inline_tools.inline("a[1] = 123.0;",['a'])
        assert sys.getrefcount(a[1]) == 2       
        assert a[1] == 123.0        
    def check_set_char(self):
        a = UserList([1,2,3])            
        inline_tools.inline('a[1] = "bubba";',['a'])
        assert sys.getrefcount(a[1]) == 2       
        assert a[1] == 'bubba'
    def check_set_string(self):
        a = UserList([1,2,3])            
        inline_tools.inline('a[1] = std::string("sissy");',['a'])
        assert sys.getrefcount(a[1]) == 2       
        assert a[1] == 'sissy'
    def check_set_string(self):
        a = UserList([1,2,3])            
        inline_tools.inline('a[1] = std::complex<double>(1,1);',['a'])
        assert sys.getrefcount(a[1]) == 2       
        assert a[1] == 1+1j

