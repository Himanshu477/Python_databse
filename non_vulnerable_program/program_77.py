import time

def print_compare(n):
    print 'Printing %d integers:'%n
    t1 = time.time()
    for i in range(n):
        print i,
    t2 = time.time()
    py = (t2-t1)
    
    # get it in cache
    inline_tools.inline('printf("%d",i);',['i'])
    t1 = time.time()
    for i in range(n):
        inline_tools.inline('printf("%d",i);',['i'])
    t2 = time.time()
    print ' speed in python:', py
    print ' speed in c:',(t2 - t1)    
    print ' speed up: %3.2f' % (py/(t2-t1))

def cout_example(lst):
    # get it in cache
    i = lst[0]
    inline_tools.inline('std::cout << i << std::endl;',['i'])
    t1 = time.time()
    for i in lst:
        inline_tools.inline('std::cout << i << std::endl;',['i'])
    t2 = time.time()
    
if __name__ == "__main__":
    n = 3000
    print_compare(n)    
    print "calling cout with integers:"
    cout_example([1,2,3])
    print "calling cout with strings:"
    cout_example(['a','bb', 'ccc'])

# This tests the amount of overhead added for inline() function calls.
# It isn't a "real world" test, but is somewhat informative.
# C:\home\ej\wrk\scipy\compiler\examples>python py_none.py
# python: 0.0199999809265
# inline: 0.160000085831
# speed up: 0.124999813736 (this is about a factor of 8 slower)

