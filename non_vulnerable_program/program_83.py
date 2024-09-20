import inline_tools

def multi_return():
    return 1, '2nd'

def c_multi_return():

    code =  """
             Py::Tuple results(2);
             results[0] = Py::Int(1);
             results[1] = Py::String("2nd");
             return_val = Py::new_reference_to(results);             
            """
    return inline_tools.inline(code,[])


def compare(m):
    import time
    t1 = time.time()
    for i in range(m):
        py_result = multi_return()
    t2 = time.time()
    py = t2 - t1
    print 'python speed:', py
    
    #load cache
    result = c_multi_return()
    t1 = time.time()
    for i in range(m):
        c_result = c_multi_return()
    t2 = time.time()
    c = t2-t1
    print 'c speed:', c
    print 'speed up:', py / c
    print 'or slow down (more likely:', c / py
    print 'python result:', py_result
    print 'c result:', c_result
    
if __name__ == "__main__":
    compare(10000)

""" 
"""
# C:\home\ej\wrk\scipy\compiler\examples>python vq.py
# vq with 1000 observation, 10 features and 30 codes fo 100 iterations
#  speed in python: 0.150119999647
# [25 29] [ 2.49147266  3.83021032]
#  speed in standard c: 0.00710999965668
# [25 29] [ 2.49147266  3.83021032]
#  speed up: 21.11
#  speed inline/blitz: 0.0186300003529
# [25 29] [ 2.49147272  3.83021021]
#  speed up: 8.06
#  speed inline/blitz2: 0.00461000084877
# [25 29] [ 2.49147272  3.83021021]
#  speed up: 32.56
 
