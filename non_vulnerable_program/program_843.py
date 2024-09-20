from auxfuncs import *
import capi_maps
import cfuncs
##############

usemodule_rules={
    'body':"""
#begintitle#
static char doc_#apiname#[] = \"\\\nVariable wrapper signature:\\n\\
\t #name# = get_#name#()\\n\\
Arguments:\\n\\
#docstr#\";
extern F_MODFUNC(#usemodulename#,#USEMODULENAME#,#realname#,#REALNAME#);
static PyObject *#apiname#(PyObject *capi_self, PyObject *capi_args) {
/*#decl#*/
\tif (!PyArg_ParseTuple(capi_args, \"\")) goto capi_fail;
printf(\"c: %d\\n\",F_MODFUNC(#usemodulename#,#USEMODULENAME#,#realname#,#REALNAME#));
\treturn Py_BuildValue(\"\");
capi_fail:
\treturn NULL;
}
""",
    'method':'\t{\"get_#name#\",#apiname#,METH_VARARGS|METH_KEYWORDS,doc_#apiname#},',
    'need':['F_MODFUNC']
    }

################

def buildusevars(m,r):
    ret={}
    outmess('\t\tBuilding use variable hooks for module "%s" (feature only for F90/F95)...\n'%(m['name']))
    varsmap={}
    revmap={}
    if r.has_key('map'):
        for k in r['map'].keys():
            if revmap.has_key(r['map'][k]):
                outmess('\t\t\tVariable "%s<=%s" is already mapped by "%s". Skipping.\n'%(r['map'][k],k,revmap[r['map'][k]]))
            else:
                revmap[r['map'][k]]=k
    if r.has_key('only') and r['only']:
        for v in r['map'].keys():
            if m['vars'].has_key(r['map'][v]):
                
                if revmap[r['map'][v]]==v:
                    varsmap[v]=r['map'][v]
                else:
                    outmess('\t\t\tIgnoring map "%s=>%s". See above.\n'%(v,r['map'][v]))
            else:
                outmess('\t\t\tNo definition for variable "%s=>%s". Skipping.\n'%(v,r['map'][v]))
    else:
        for v in m['vars'].keys():
            if revmap.has_key(v):
                varsmap[v]=revmap[v]
            else:
                varsmap[v]=v
    for v in varsmap.keys():
        ret=dictappend(ret,buildusevar(v,varsmap[v],m['vars'],m['name']))
    return ret
def buildusevar(name,realname,vars,usemodulename):
    outmess('\t\t\tConstructing wrapper function for variable "%s=>%s"...\n'%(name,realname))
    ret={}
    vrd={'name':name,
         'realname':realname,
         'REALNAME':string.upper(realname),
         'usemodulename':usemodulename,
         'USEMODULENAME':string.upper(usemodulename),
         'texname':string.replace(name,'_','\\_'),
         'begintitle':gentitle('%s=>%s'%(name,realname)),
         'endtitle':gentitle('end of %s=>%s'%(name,realname)),
         'apiname':'#modulename#_use_%s_from_%s'%(realname,usemodulename)
         }
    nummap={0:'Ro',1:'Ri',2:'Rii',3:'Riii',4:'Riv',5:'Rv',6:'Rvi',7:'Rvii',8:'Rviii',9:'Rix'}
    vrd['texnamename']=name
    for i in nummap.keys():
        vrd['texnamename']=string.replace(vrd['texnamename'],`i`,nummap[i])
    if hasnote(vars[realname]): vrd['note']=vars[realname]['note']
    rd=dictappend({},vrd)
    var=vars[realname]
    
    print name,realname,vars[realname]
    ret=applyrules(usemodule_rules,rd)
    return ret








# mtrand.pyx -- A Pyrex wrapper of Jean-Sebastien Roy's RandomKit
#
# Copyright 2005 Robert Kern (robert.kern@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

include "Python.pxi"
include "Numeric.pxi"

cdef extern from "math.h":
    double exp(double x)
    double log(double x)
    double floor(double x)
    double sin(double x)
    double cos(double x)

cdef extern from "randomkit.h":

    ctypedef struct rk_state:
        unsigned long key[624]
        int pos

    ctypedef enum rk_error:
        RK_NOERR = 0
        RK_ENODEV = 1
        RK_ERR_MAX = 2

    char *rk_strerror[2]

    # 0xFFFFFFFFUL
    unsigned long RK_MAX

    void rk_seed(unsigned long seed, rk_state *state)
    rk_error rk_randomseed(rk_state *state)
    unsigned long rk_random(rk_state *state)
    long rk_long(rk_state *state)
    unsigned long rk_ulong(rk_state *state)
    unsigned long rk_interval(unsigned long max, rk_state *state)
    double rk_double(rk_state *state)
    void rk_fill(void *buffer, size_t size, rk_state *state)
    rk_error rk_devfill(void *buffer, size_t size, int strong)
    rk_error rk_altfill(void *buffer, size_t size, int strong,
            rk_state *state)
    double rk_gauss(rk_state *state)

cdef extern from "distributions.h":
    
    double rk_normal(rk_state *state, double loc, double scale)
    double rk_standard_exponential(rk_state *state)
    double rk_exponential(rk_state *state, double scale)
    double rk_uniform(rk_state *state, double loc, double scale)
    double rk_standard_gamma(rk_state *state, double shape)
    double rk_gamma(rk_state *state, double shape, double scale)
    double rk_beta(rk_state *state, double a, double b)
    double rk_chisquare(rk_state *state, double df)
    double rk_noncentral_chisquare(rk_state *state, double df, double nonc)
    double rk_f(rk_state *state, double dfnum, double dfden)
    double rk_noncentral_f(rk_state *state, double dfnum, double dfden, double nonc)
    double rk_standard_cauchy(rk_state *state)
    double rk_standard_t(rk_state *state, double df)
    double rk_vonmises(rk_state *state, double mu, double kappa)
    double rk_pareto(rk_state *state, double a)
    double rk_weibull(rk_state *state, double a)
    double rk_power(rk_state *state, double a)
    long rk_binomial(rk_state *state, long n, double p)
    long rk_negative_binomial(rk_state *state, long n, double p)
    long rk_poisson(rk_state *state, double lam)

ctypedef double (* rk_cont0)(rk_state *state)
ctypedef double (* rk_cont1)(rk_state *state, double a)
ctypedef double (* rk_cont2)(rk_state *state, double a, double b)
ctypedef double (* rk_cont3)(rk_state *state, double a, double b, double c)

ctypedef long (* rk_disc0)(rk_state *state)
ctypedef long (* rk_discnp)(rk_state *state, long n, double p)
ctypedef long (* rk_discd)(rk_state *state, double a)


cdef extern from "initarray.h":
   void init_by_array(rk_state *self, unsigned long *init_key, 
                      unsigned long key_length)

# Initialize Numeric
