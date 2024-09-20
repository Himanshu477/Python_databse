    import re

routine_start_re = re.compile(r'(\n|\A)((     (\$|\*))|)\s*(subroutine|function)\b',re.I)
routine_end_re = re.compile(r'\n\s*end\s*(subroutine|function)\b.*(\n|\Z)',re.I)
function_start_re = re.compile(r'\n     (\$|\*)\s*function\b',re.I)

def parse_structure(astr):
    """ Return a list of tuples for each function or subroutine each
    tuple is the start and end of a subroutine or function to be
    expanded.
    """

    spanlist = []
    ind = 0
    while 1:
        m = routine_start_re.search(astr,ind)
        if m is None:
            break
        start = m.start()
        if function_start_re.match(astr,start,m.end()):
            while 1:
                i = astr.rfind('\n',ind,start)
                if i==-1:
                    break
                start = i
                if astr[i:i+7]!='\n     $':
                    break
        start += 1
        m = routine_end_re.search(astr,m.end())
        ind = end = m and m.end()-1 or len(astr)
        spanlist.append((start,end))
    return spanlist

template_re = re.compile(r"<\s*(\w[\w\d]*)\s*>")
named_re = re.compile(r"<\s*(\w[\w\d]*)\s*=\s*(.*?)\s*>")
list_re = re.compile(r"<\s*((.*?))\s*>")

def find_repl_patterns(astr):
    reps = named_re.findall(astr)
    names = {}
    for rep in reps:
        name = rep[0].strip() or unique_key(names)
        repl = rep[1].replace('\,','@comma@')
        thelist = conv(repl)
        names[name] = thelist
    return names

item_re = re.compile(r"\A\\(?P<index>\d+)\Z")
def conv(astr):
    b = astr.split(',')
    l = [x.strip() for x in b]
    for i in range(len(l)):
        m = item_re.match(l[i])
        if m:
            j = int(m.group('index'))
            l[i] = l[j]
    return ','.join(l)

def unique_key(adict):
    """ Obtain a unique key given a dictionary."""
    allkeys = adict.keys()
    done = False
    n = 1
    while not done:
        newkey = '__l%s' % (n)
        if newkey in allkeys:
            n += 1
        else:
            done = True
    return newkey


template_name_re = re.compile(r'\A\s*(\w[\w\d]*)\s*\Z')
def expand_sub(substr,names):
    substr = substr.replace('\>','@rightarrow@')
    substr = substr.replace('\<','@leftarrow@')
    lnames = find_repl_patterns(substr)
    substr = named_re.sub(r"<\1>",substr)  # get rid of definition templates

    def listrepl(mobj):
        thelist = conv(mobj.group(1).replace('\,','@comma@'))
        if template_name_re.match(thelist):
            return "<%s>" % (thelist)
        name = None
        for key in lnames.keys():    # see if list is already in dictionary
            if lnames[key] == thelist:
                name = key
        if name is None:      # this list is not in the dictionary yet
            name = unique_key(lnames)
            lnames[name] = thelist
        return "<%s>" % name

    substr = list_re.sub(listrepl, substr) # convert all lists to named templates
                                           # newnames are constructed as needed

    numsubs = None
    base_rule = None
    rules = {}
    for r in template_re.findall(substr):
        if not rules.has_key(r):
            thelist = lnames.get(r,names.get(r,None))
            if thelist is None:
                raise ValueError,'No replicates found for <%s>' % (r)
            if not names.has_key(r) and not thelist.startswith('_'):
                names[r] = thelist
            rule = [i.replace('@comma@',',') for i in thelist.split(',')]
            num = len(rule)

            if numsubs is None:
                numsubs = num
                rules[r] = rule
                base_rule = r
            elif num == numsubs:
                rules[r] = rule
            else:
                print "Mismatch in number of replacements (base <%s=%s>)"\
                      " for <%s=%s>. Ignoring." % (base_rule,
                                                  ','.join(rules[base_rule]),
                                                  r,thelist)
    if not rules:
        return substr

    def namerepl(mobj):
        name = mobj.group(1)
        return rules.get(name,(k+1)*[name])[k]

    newstr = ''
    for k in range(numsubs):
        newstr += template_re.sub(namerepl, substr) + '\n\n'

    newstr = newstr.replace('@rightarrow@','>')
    newstr = newstr.replace('@leftarrow@','<')
    return newstr
    
def process_str(allstr):
    newstr = allstr
    writestr = '' #_head # using _head will break free-format files

    struct = parse_structure(newstr)

    oldend = 0
    names = {}
    names.update(_special_names)
    for sub in struct:
        writestr += newstr[oldend:sub[0]]
        names.update(find_repl_patterns(newstr[oldend:sub[0]]))
        writestr += expand_sub(newstr[sub[0]:sub[1]],names)
        oldend =  sub[1]
    writestr += newstr[oldend:]

    return writestr

include_src_re = re.compile(r"(\n|\A)\s*include\s*['\"](?P<name>[\w\d./\\]+[.]src)['\"]",re.I)

def resolve_includes(source):
    d = os.path.dirname(source)
    fid = open(source)
    lines = []
    for line in fid.readlines():
        m = include_src_re.match(line)
        if m:
            fn = m.group('name')
            if not os.path.isabs(fn):
                fn = os.path.join(d,fn)
            if os.path.isfile(fn):
                print 'Including file',fn
                lines.extend(resolve_includes(fn))
            else:
                lines.append(line)
        else:
            lines.append(line)
    fid.close()
    return lines

def process_file(source):
    lines = resolve_includes(source)
    return process_str(''.join(lines))

_special_names = find_repl_patterns('''
<_c=s,d,c,z>
<_t=real,double precision,complex,double complex>
<prefix=s,d,c,z>
<ftype=real,double precision,complex,double complex>
<ctype=float,double,complex_float,complex_double>
<ftypereal=real,double precision,\\0,\\1>
<ctypereal=float,double,\\0,\\1>
''')

if __name__ == "__main__":

    try:
        file = sys.argv[1]
    except IndexError:
        fid = sys.stdin
        outfile = sys.stdout
    else:
        fid = open(file,'r')
        (base, ext) = os.path.splitext(file)
        newname = base
        outfile = open(newname,'w')

    allstr = fid.read()
    writestr = process_str(allstr)
    outfile.write(writestr)


""" Functions for converting from DOS to UNIX line endings
"""

