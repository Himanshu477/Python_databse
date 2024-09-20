from inline_tools import inline

def py_func():
    return None

n = 10000
t1 = time.time()    
for i in range(n):
    py_func()
t2 = time.time()
py_time = t2 - t1
print 'python:', py_time    

inline("",[])
t1 = time.time()    
for i in range(n):
    inline("",[])
t2 = time.time()
print 'inline:', (t2-t1)    
print 'speed up:', py_time/(t2-t1)    
print 'or (more likely) slow down:', (t2-t1)/py_time

