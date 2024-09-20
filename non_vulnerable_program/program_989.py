from UserList import UserList
obj = UserList([1,[1,2],"hello"])
code = """
       int i;
       // find obj length and accesss each of its items
       std::cout << "UserList items: ";
       for(i = 0; i < obj.length(); i++)
           std::cout << obj[i] << " ";
       std::cout << std::endl;
       // assign new values to each of its items
       for(i = 0; i < obj.length(); i++)
           obj[i] = "goodbye";
       """
weave.inline(code,['obj'])       
print "obj with new values:", obj


""" Comparison of several different ways of calculating a "ramp"
    function.
    
    C:\home\ej\wrk\junk\scipy\weave\examples>python ramp.py 
    python (seconds*ratio): 128.149998188
    arr[500]: 0.0500050005001
    compiled numeric1 (seconds, speed up): 1.42199993134 90.1195530071
    arr[500]: 0.0500050005001
    compiled numeric2 (seconds, speed up): 0.950999975204 134.752893301
    arr[500]: 0.0500050005001
    compiled list1 (seconds, speed up): 53.100001812 2.41337088164
    arr[500]: 0.0500050005001
    compiled list4 (seconds, speed up): 30.5500030518 4.19476220578
    arr[500]: 0.0500050005001     

"""

