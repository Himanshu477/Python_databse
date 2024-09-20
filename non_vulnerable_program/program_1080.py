from UserDict import UserDict
class test_object_set_item_op_key(ScipyTestCase):
    def check_key_refcount(self,level=5):
        a = UserDict()
        code =  """
                py::object one = 1;
                py::object two = 2;
                py::tuple ref_counts(3);
                py::tuple obj_counts(3);
                py::tuple val_counts(3);
                py::tuple key_counts(3);
                obj_counts[0] = a.refcount();
                key_counts[0] = one.refcount();
                val_counts[0] = two.refcount();
                a[1] = 2;
                obj_counts[1] = a.refcount();
                key_counts[1] = one.refcount();
                val_counts[1] = two.refcount();
                a[1] = 2;
                obj_counts[2] = a.refcount();
                key_counts[2] = one.refcount();
                val_counts[2] = two.refcount();
                
                ref_counts[0] = obj_counts;
                ref_counts[1] = key_counts;
                ref_counts[2] = val_counts;
                return_val = ref_counts;
                """
        obj,key,val = inline_tools.inline(code,['a'])
        assert obj[0] == obj[1] and obj[1] == obj[2]
        assert key[0] + 1 == key[1] and key[1] == key[2]
        assert val[0] + 1 == val[1] and val[1] == val[2]
        
    def check_set_double_exists(self,level=5):
        a = UserDict()   
        key = 10.0     
        a[key] = 100.0
        inline_tools.inline('a[key] = 123.0;',['a','key'])
        first = sys.getrefcount(key)
        inline_tools.inline('a[key] = 123.0;',['a','key'])
        second = sys.getrefcount(key)
        assert first == second
        # !! I think the following should be 3
        assert sys.getrefcount(key) ==  5
        assert sys.getrefcount(a[key]) == 2       
        assert a[key] == 123.0
    def check_set_double_new(self,level=5):
        a = UserDict()        
        key = 1.0
        inline_tools.inline('a[key] = 123.0;',['a','key'])
        assert sys.getrefcount(key) == 4 # should be 3       
        assert sys.getrefcount(a[key]) == 2       
        assert a[key] == 123.0
    def check_set_complex(self,level=5):
        a = UserDict()
        key = 1+1j            
        inline_tools.inline("a[key] = 1234;",['a','key'])
        assert sys.getrefcount(key) == 3       
        assert sys.getrefcount(a[key]) == 2                
        assert a[key] == 1234
    def check_set_char(self,level=5):
        a = UserDict()        
        inline_tools.inline('a["hello"] = 123.0;',['a'])
        assert sys.getrefcount(a["hello"]) == 2       
        assert a["hello"] == 123.0
        
    def check_set_class(self,level=5):
        a = UserDict()        
        class foo:
            def __init__(self,val):
                self.val = val
            def __hash__(self):
                return self.val
        key = foo(4)
        inline_tools.inline('a[key] = "bubba";',['a','key'])
        first = sys.getrefcount(key)       
        inline_tools.inline('a[key] = "bubba";',['a','key'])
        second = sys.getrefcount(key)       
        # I don't think we're leaking if this is true
        assert first == second  
        # !! BUT -- I think this should be 3
        assert sys.getrefcount(key) == 4 
        assert sys.getrefcount(a[key]) == 2       
        assert a[key] == 'bubba'
    def check_set_from_member(self,level=5):
        a = UserDict()        
        a['first'] = 1
        a['second'] = 2
        inline_tools.inline('a["first"] = a["second"];',['a'])
        assert a['first'] == a['second']

if __name__ == "__main__":
    ScipyTest().run()


""" Test refcounting and behavior of SCXX.
"""

