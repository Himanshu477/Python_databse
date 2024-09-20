import scipy.weave as weave

#----------------------------------------------------------------------------
# get/set attribute and call methods example
#----------------------------------------------------------------------------

class foo:
    def __init__(self):
        self.val = 1
    def inc(self,amount):
        self.val += 1
        return self.val
obj = foo()
code = """
       int i = obj.attr("val");
       std::cout << "initial val: " << i << std::endl;
       
       py::tuple args(1);
       args[0] = 2; 
       i = obj.mcall("inc",args);
       std::cout << "inc result: " << i << std::endl;
       
       obj.set_attr("val",5);
       i = obj.attr("val");
       std::cout << "after set attr: " << i << std::endl;
       """
weave.inline(code,['obj'])       
       
#----------------------------------------------------------------------------
# indexing of values.
#----------------------------------------------------------------------------
