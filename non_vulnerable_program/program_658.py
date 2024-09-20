import re
_poly_mat = re.compile(r"[*][*]([0-9]*)")
def _raise_power(astr, wrap=70):
    n = 0
    line1 = ''
    line2 = ''
    output = ' '
    while 1:
        mat = _poly_mat.search(astr,n)
        if mat is None:
            break
        span = mat.span()
        power = mat.groups()[0]
        partstr = astr[n:span[0]]
        n = span[1]
        toadd2 = partstr + ' '*(len(power)-1)
        toadd1 = ' '*(len(partstr)-1) + power
        if ((len(line2)+len(toadd2) > wrap) or \
            (len(line1)+len(toadd1) > wrap)):
            output += line1 + "\n" + line2 + "\n "
            line1 = toadd1
            line2 = toadd2
        else:                
            line2 += partstr + ' '*(len(power)-1)
            line1 += ' '*(len(partstr)-1) + power
    output += line1 + "\n" + line2
    return output + astr[n:]
    
                       
class poly1d:
    """A one-dimensional polynomial class.

    p = poly1d([1,2,3]) constructs the polynomial x**2 + 2 x + 3

    p(0.5) evaluates the polynomial at the location
    p.r  is a list of roots
    p.c  is the coefficient array [1,2,3]
    p.order is the polynomial order (after leading zeros in p.c are removed)
    p[k] is the coefficient on the kth power of x (backwards from
         sequencing the coefficient array.

    polynomials can be added, substracted, multplied and divided (returns
         quotient and remainder).
    asarray(p) will also give the coefficient array, so polynomials can
         be used in all functions that accept arrays.
    """
    def __init__(self, c_or_r, r=0):
        if isinstance(c_or_r,poly1d):
            for key in c_or_r.__dict__.keys():
                self.__dict__[key] = c_or_r.__dict__[key]
            return
        if r:
            c_or_r = poly(c_or_r)
        c_or_r = atleast_1d(c_or_r)
        if len(c_or_r.shape) > 1:
            raise ValueError, "Polynomial must be 1d only."
        c_or_r = trim_zeros(c_or_r, trim='f')
        if len(c_or_r) == 0:
            c_or_r = _nx.array([0])
        self.__dict__['coeffs'] = c_or_r
        self.__dict__['order'] = len(c_or_r) - 1

    def __array__(self,t=None):
        if t:
            return asarray(self.coeffs,t)
        else:
            return asarray(self.coeffs)

    def __coerce__(self,other):
        return None
    
    def __repr__(self):
        vals = repr(self.coeffs)
        vals = vals[6:-1]
        return "poly1d(%s)" % vals

    def __len__(self):
        return self.order

    def __str__(self):
        N = self.order
        thestr = "0"
        for k in range(len(self.coeffs)):
            coefstr ='%.4g' % abs(self.coeffs[k])
            if coefstr[-4:] == '0000':
                coefstr = coefstr[:-5]
            power = (N-k)
            if power == 0:
                if coefstr != '0':
                    newstr = '%s' % (coefstr,)
                else:
                    if k == 0:
                        newstr = '0'
                    else:
                        newstr = ''
            elif power == 1:
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = 'x'
                else:                    
                    newstr = '%s x' % (coefstr,)
            else:
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = 'x**%d' % (power,)
                else:                    
                    newstr = '%s x**%d' % (coefstr, power)

            if k > 0:
                if newstr != '':
                    if self.coeffs[k] < 0:
                        thestr = "%s - %s" % (thestr, newstr)
                    else:
                        thestr = "%s + %s" % (thestr, newstr)
            elif (k == 0) and (newstr != '') and (self.coeffs[k] < 0):
                thestr = "-%s" % (newstr,)
            else:
                thestr = newstr
        return _raise_power(thestr)
        

    def __call__(self, val):
        return polyval(self.coeffs, val)

    def __mul__(self, other):
        if isscalar(other):
            return poly1d(self.coeffs * other)
        else:
            other = poly1d(other)
            return poly1d(polymul(self.coeffs, other.coeffs))

    def __rmul__(self, other):
        if isscalar(other):
            return poly1d(other * self.coeffs)
        else:
            other = poly1d(other)
            return poly1d(polymul(self.coeffs, other.coeffs))        
    
    def __add__(self, other):
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))        
        
    def __radd__(self, other):
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __pow__(self, val):
        if not isscalar(val) or int(val) != val or val < 0:
            raise ValueError, "Power to non-negative integers only."
        res = [1]
        for k in range(val):
            res = polymul(self.coeffs, res)
        return poly1d(res)

    def __sub__(self, other):
        other = poly1d(other)
        return poly1d(polysub(self.coeffs, other.coeffs))

    def __rsub__(self, other):
        other = poly1d(other)
        return poly1d(polysub(other.coeffs, self.coeffs))

    def __div__(self, other):
        if isscalar(other):
            return poly1d(self.coeffs/other)
        else:
            other = poly1d(other)
            return map(poly1d,polydiv(self.coeffs, other.coeffs))

    def __rdiv__(self, other):
        if isscalar(other):
            return poly1d(other/self.coeffs)
        else:
            other = poly1d(other)
            return map(poly1d,polydiv(other.coeffs, self.coeffs))

    def __setattr__(self, key, val):
        raise ValueError, "Attributes cannot be changed this way."

    def __getattr__(self, key):
        if key in ['r','roots']:
            return roots(self.coeffs)
        elif key in ['S1','coef','coefficients']:
            return self.coeffs
        elif key in ['o']:
            return self.order
        else:
            return self.__dict__[key]
        
    def __getitem__(self, val):
        ind = self.order - val
        if val > self.order:
            return 0
        if val < 0:
            return 0
        return self.coeffs[ind]

    def __setitem__(self, key, val):
        ind = self.order - key
        if key < 0:
            raise ValueError, "Does not support negative powers."
        if key > self.order:
            zr = _nx.zeros(key-self.order,self.coeffs.dtypechar)
            self.__dict__['coeffs'] = _nx.concatenate((zr,self.coeffs))
            self.__dict__['order'] = key
            ind = 0
        self.__dict__['coeffs'][ind] = val
        return

    def integ(self, m=1, k=0):
        return poly1d(polyint(self.coeffs,m=m,k=k))

    def deriv(self, m=1):
        return poly1d(polyder(self.coeffs,m=m))


## Automatically adapted for scipy Sep 19, 2005 by convertcode.py

"""
Wrapper functions to more user-friendly calling of certain math functions
whose output is different than the input in certain domains of the input.
"""

__all__ = ['sqrt', 'log', 'log2','logn','log10', 'power', 'arccos',
           'arcsin', 'arctanh']

