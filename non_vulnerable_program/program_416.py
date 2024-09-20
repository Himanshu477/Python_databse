from fnmatch import fnmatch

def filter_machine(x,filt):
    return (fnmatch(x.machine,filt) or fnmatch(x.long_machine,filt))
def filter_user(x,filt):
    return (fnmatch(x.user,filt))   
def filter_state(x,filt):
    return (fnmatch(x.state,filt))
def filter_command(x,filt):
    return (fnmatch(x.cmdline,filt))
def filter_cpu(x,filt):
    return eval(str(x.cpu_percent) + filt)
def filter_memory(x,filt):
    return eval(str(x.memory_percent) + filt)
def filter_mb(x,filt):
    return eval(str(x.total_memory) + filt)
    
ps_filter={}
ps_filter['user'] = filter_user
ps_filter['machine'] = filter_machine
ps_filter['state'] = filter_state
ps_filter['command'] = filter_command
ps_filter['memory'] = filter_memory
ps_filter['mb'] = filter_mb
ps_filter['cpu'] = filter_cpu

def ps(sort_by='cpu',**filters):
    psl = ps_list(sort_by,**filters)
    if len(psl):
        print psl[0].labels_with_name()
    for i in psl: 
        print i
    
def ps_sort(psl,sort_by='cpu',**filters):
    for f in filters.keys():
        try:
            filt = ps_filter[f]
            filt_str = filters[f]
            psl = filter(lambda x,filt=filt,y=filt_str:filt(x,y),psl)
        except KeyError:
            print 'warning: "', f, '"is an invalid key for filtering command.'
            print '         ', 'use one of the following:', str(ps_filter.keys())
    try:
        compare = ps_cmp[sort_by]    
        psl.sort(compare)
    except KeyError:
        print 'warning: "', sort_by, '"is an invalid choice for sorting.'
        print '         ', 'use one of the following:', str(ps_cmp.keys())
    return psl




#!/usr/bin/env python
