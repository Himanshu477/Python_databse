import random, md5, time, cStringIO

def speed(n,m):
    s = 'a'*n
    t1 = time.time()            
    for i in range(m):
        q= md5.new(s).digest()
    t2 = time.time()
    print (t2 - t1) / m

#speed(50,1e6)

def generate_random(avg_length,count):
    all_str = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    lo,hi = [30,avg_length*2+30]
    for i in range(count):
        new_str = cStringIO.StringIO()
        l = random.randrange(lo,hi)
        for i in range(l):
            new_str.write(random.choice(alphabet))
        all_str.append(new_str.getvalue())
    return all_str    
    
def md5_dict(lst):
    catalog = {}
    t1 = time.time()
    for s in lst:
        key= md5.new(s).digest()
        catalog[key] = None
    t2 = time.time()    
    print 'md5 build(len,sec,per):', len(lst), t2 - t1, (t2-t1)/len(lst)
    
    t1 = time.time()
    for s in lst:
        key= md5.new(s).digest()
        val = catalog[key]
    t2 = time.time()    
    print 'md5 retrv(len,sec,per):', len(lst), t2 - t1, (t2-t1)/len(lst)

def std_dict(lst):
    catalog = {}
    t1 = time.time()
    for s in lst:
        catalog[s] = None
    t2 = time.time()    
    print 'std build(len,sec,per):', len(lst), t2 - t1, (t2-t1)/len(lst)
    
    t1 = time.time()
    for s in lst:
        val = catalog[s]
    t2 = time.time()    
    print 'std retrv(len,sec,per):', len(lst), t2 - t1, (t2-t1)/len(lst)

def run(m=200,n=10):
    lst = generate_random(m,n)
    md5_dict(lst)
    std_dict(lst)

run()    

