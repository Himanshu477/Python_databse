        from 1.0, gives something different from 1.0.
    epsneg : float
        Floating-point number ``beta**negep``.
    iexp : int
        Number of bits in the exponent (including its sign and bias).
    minexp : int
        Smallest (most negative) power of `ibeta` consistent with there
        being no leading zeros in the mantissa.
    xmin : float
        Floating point number ``beta**minexp`` (the smallest [in
        magnitude] usable floating value).
    maxexp : int
        Smallest (positive) power of `ibeta` that causes overflow.
    xmax : float
        ``(1-epsneg) * beta**maxexp`` (the largest [in magnitude]
        usable floating value).
    irnd : int
        In ``range(6)``, information on what kind of rounding is done
        in addition, and on how underflow is handled.
    ngrd : int
        Number of 'guard digits' used when truncating the product
        of two mantissas to fit the representation.
    epsilon : float
        Same as `eps`.
    tiny : float
        Same as `xmin`.
    huge : float
        Same as `xmax`.
    precision : float
        ``- int(-log10(eps))``
    resolution : float
        ``- 10**(-precision)``

    Parameters
    ----------
    float_conv : function, optional
        Function that converts an integer or integer array to a float
        or float array. Default is `float`.
    int_conv : function, optional
        Function that converts a float or float array to an integer or
        integer array. Default is `int`.
    float_to_float : function, optional
        Function that converts a float array to float. Default is `float`.
        Note that this does not seem to do anything useful in the current
        implementation.
    float_to_str : function, optional
        Function that converts a single float to a string. Default is
        ``lambda v:'%24.16e' %v``.
    title : str, optional
        Title that is printed in the string representation of `MachAr`.

    See Also
    --------
    finfo : Machine limits for floating point types.
    iinfo : Machine limits for integer types.

    References
    ----------
    .. [1] Press, Teukolsky, Vetterling and Flannery,
           "Numerical Recipes in C++," 2nd ed,
           Cambridge University Press, 2002, p. 31.

    """
    def __init__(self, float_conv=float,int_conv=int,
                 float_to_float=float,
                 float_to_str = lambda v:'%24.16e' % v,
                 title = 'Python floating point number'):
        """
          float_conv - convert integer to float (array)
          int_conv   - convert float (array) to integer
          float_to_float - convert float array to float
          float_to_str - convert array float to str
          title        - description of used floating point numbers
        """
        # We ignore all errors here because we are purposely triggering
        # underflow to detect the properties of the runninng arch.
        saverrstate = seterr(under='ignore')
        try:
            self._do_init(float_conv, int_conv, float_to_float, float_to_str, title)
        finally:
            seterr(**saverrstate)

    def _do_init(self, float_conv, int_conv, float_to_float, float_to_str, title):
        max_iterN = 10000
        msg = "Did not converge after %d tries with %s"
        one = float_conv(1)
        two = one + one
        zero = one - one

        # Do we really need to do this?  Aren't they 2 and 2.0?
        # Determine ibeta and beta
        a = one
        for _ in xrange(max_iterN):
            a = a + a
            temp = a + one
            temp1 = temp - a
            if any(temp1 - one != zero):
                break
        else:
            raise RuntimeError, msg % (_, one.dtype)
        b = one
        for _ in xrange(max_iterN):
            b = b + b
            temp = a + b
            itemp = int_conv(temp-a)
            if any(itemp != 0):
                break
        else:
            raise RuntimeError, msg % (_, one.dtype)
        ibeta = itemp
        beta = float_conv(ibeta)

        # Determine it and irnd
        it = -1
        b = one
        for _ in xrange(max_iterN):
            it = it + 1
            b = b * beta
            temp = b + one
            temp1 = temp - b
            if any(temp1 - one != zero):
                break
        else:
            raise RuntimeError, msg % (_, one.dtype)

        betah = beta / two
        a = one
        for _ in xrange(max_iterN):
            a = a + a
            temp = a + one
            temp1 = temp - a
            if any(temp1 - one != zero):
                break
        else:
            raise RuntimeError, msg % (_, one.dtype)
        temp = a + betah
        irnd = 0
        if any(temp-a != zero):
            irnd = 1
        tempa = a + beta
        temp = tempa + betah
        if irnd==0 and any(temp-tempa != zero):
            irnd = 2

        # Determine negep and epsneg
        negep = it + 3
        betain = one / beta
        a = one
        for i in range(negep):
            a = a * betain
        b = a
        for _ in xrange(max_iterN):
            temp = one - a
            if any(temp-one != zero):
                break
            a = a * beta
            negep = negep - 1
            # Prevent infinite loop on PPC with gcc 4.0:
            if negep < 0:
                raise RuntimeError, "could not determine machine tolerance " \
                                    "for 'negep', locals() -> %s" % (locals())
        else:
            raise RuntimeError, msg % (_, one.dtype)
        negep = -negep
        epsneg = a

        # Determine machep and eps
        machep = - it - 3
        a = b

        for _ in xrange(max_iterN):
            temp = one + a
            if any(temp-one != zero):
                break
            a = a * beta
            machep = machep + 1
        else:
            raise RuntimeError, msg % (_, one.dtype)
        eps = a

        # Determine ngrd
        ngrd = 0
        temp = one + eps
        if irnd==0 and any(temp*one - one != zero):
            ngrd = 1

        # Determine iexp
        i = 0
        k = 1
        z = betain
        t = one + eps
        nxres = 0
        for _ in xrange(max_iterN):
            y = z
            z = y*y
            a = z*one # Check here for underflow
            temp = z*t
            if any(a+a == zero) or any(abs(z)>=y):
                break
            temp1 = temp * betain
            if any(temp1*beta == z):
                break
            i = i + 1
            k = k + k
        else:
            raise RuntimeError, msg % (_, one.dtype)
        if ibeta != 10:
            iexp = i + 1
            mx = k + k
        else:
            iexp = 2
            iz = ibeta
            while k >= iz:
                iz = iz * ibeta
                iexp = iexp + 1
            mx = iz + iz - 1

        # Determine minexp and xmin
        for _ in xrange(max_iterN):
            xmin = y
            y = y * betain
            a = y * one
            temp = y * t
            if any(a+a != zero) and any(abs(y) < xmin):
                k = k + 1
                temp1 = temp * betain
                if any(temp1*beta == y) and any(temp != y):
                    nxres = 3
                    xmin = y
                    break
            else:
                break
        else:
            raise RuntimeError, msg % (_, one.dtype)
        minexp = -k

        # Determine maxexp, xmax
        if mx <= k + k - 3 and ibeta != 10:
            mx = mx + mx
            iexp = iexp + 1
        maxexp = mx + minexp
        irnd = irnd + nxres
        if irnd >= 2:
            maxexp = maxexp - 2
        i = maxexp + minexp
        if ibeta == 2 and not i:
            maxexp = maxexp - 1
        if i > 20:
            maxexp = maxexp - 1
        if any(a != y):
            maxexp = maxexp - 2
        xmax = one - epsneg
        if any(xmax*one != xmax):
            xmax = one - beta*epsneg
        xmax = xmax / (xmin*beta*beta*beta)
        i = maxexp + minexp + 3
        for j in range(i):
            if ibeta==2:
                xmax = xmax + xmax
            else:
                xmax = xmax * beta

        self.ibeta = ibeta
        self.it = it
        self.negep = negep
        self.epsneg = float_to_float(epsneg)
        self._str_epsneg = float_to_str(epsneg)
        self.machep = machep
        self.eps = float_to_float(eps)
        self._str_eps = float_to_str(eps)
        self.ngrd = ngrd
        self.iexp = iexp
        self.minexp = minexp
        self.xmin = float_to_float(xmin)
        self._str_xmin = float_to_str(xmin)
        self.maxexp = maxexp
        self.xmax = float_to_float(xmax)
        self._str_xmax = float_to_str(xmax)
        self.irnd = irnd

        self.title = title
        # Commonly used parameters
        self.epsilon = self.eps
        self.tiny = self.xmin
        self.huge = self.xmax

        import math
        self.precision = int(-math.log10(float_to_float(self.eps)))
        ten = two + two + two + two + two
        resolution = ten ** (-self.precision)
        self.resolution = float_to_float(resolution)
        self._str_resolution = float_to_str(resolution)

    def __str__(self):
        return '''\
Machine parameters for %(title)s
---------------------------------------------------------------------
ibeta=%(ibeta)s it=%(it)s iexp=%(iexp)s ngrd=%(ngrd)s irnd=%(irnd)s
machep=%(machep)s     eps=%(_str_eps)s (beta**machep == epsilon)
negep =%(negep)s  epsneg=%(_str_epsneg)s (beta**epsneg)
minexp=%(minexp)s   xmin=%(_str_xmin)s (beta**minexp == tiny)
maxexp=%(maxexp)s    xmax=%(_str_xmax)s ((1-epsneg)*beta**maxexp == huge)
---------------------------------------------------------------------
''' % self.__dict__


if __name__ == '__main__':
    print MachAr()


'''

=============================
 Byteswapping and byte order
=============================

Introduction to byte ordering and ndarrays
==========================================

``ndarrays`` are objects that provide a python array interface to data
in memory.

It often happens that the memory that you want to view with an array is
not of the same byte ordering as the computer on which you are running
Python.

For example, I might be working on a computer with a little-endian CPU -
such as an Intel Pentium, but I have loaded some data from a file
written by a computer that is big-endian.  Let's say I have loaded 4
bytes from a file written by a Sun (big-endian) computer.  I know that
these 4 bytes represent two 16-bit integers.  On a big-endian machine, a
two-byte integer is stored with the Most Significant Byte (MSB) first,
and then the Least Significant Byte (LSB). Thus the bytes are, in memory order:

#. MSB integer 1
#. LSB integer 1
#. MSB integer 2
#. LSB integer 2

Let's say the two integers were in fact 1 and 770.  Because 770 = 256 *
3 + 2, the 4 bytes in memory would contain respectively: 0, 1, 3, 2.
The bytes I have loaded from the file would have these contents:

>>> big_end_str = chr(0) + chr(1) + chr(3) + chr(2)
>>> big_end_str
'\x00\x01\x03\x02'

We might want to use an ``ndarray`` to access these integers.  In that
case, we can create an array around this memory, and tell numpy that
there are two integers, and that they are 16 bit and big-endian:

>>> import numpy as np
>>> big_end_arr = np.ndarray(shape=(2,),dtype='>i2', buffer=big_end_str)
>>> big_end_arr[0]
1
>>> big_end_arr[1]
770

Note the array ``dtype`` above of ``>i2``.  The ``>`` means 'big-endian'
(``<`` is little-endian) and ``i2`` means 'signed 2-byte integer'.  For
example, if our data represented a single unsigned 4-byte little-endian
integer, the dtype string would be ``<u4``.

In fact, why don't we try that?

>>> little_end_u4 = np.ndarray(shape=(1,),dtype='<u4', buffer=big_end_str)
>>> little_end_u4[0] == 1 * 256**1 + 3 * 256**2 + 2 * 256**3
True

Returning to our ``big_end_arr`` - in this case our underlying data is
big-endian (data endianness) and we've set the dtype to match (the dtype
is also big-endian).  However, sometimes you need to flip these around.

Changing byte ordering
======================

As you can imagine from the introduction, there are two ways you can
affect the relationship between the byte ordering of the array and the
underlying memory it is looking at:

* Change the byte-ordering information in the array dtype so that it
  interprets the undelying data as being in a different byte order.
  This is the role of ``arr.newbyteorder()``
* Change the byte-ordering of the underlying data, leaving the dtype
  interpretation as it was.  This is what ``arr.byteswap()`` does.

The common situations in which you need to change byte ordering are:

#. Your data and dtype endianess don't match, and you want to change
   the dtype so that it matches the data.
#. Your data and dtype endianess don't match, and you want to swap the
   data so that they match the dtype
#. Your data and dtype endianess match, but you want the data swapped
   and the dtype to reflect this

Data and dtype endianness don't match, change dtype to match data
-----------------------------------------------------------------

We make something where they don't match:

>>> wrong_end_dtype_arr = np.ndarray(shape=(2,),dtype='<i2', buffer=big_end_str)
>>> wrong_end_dtype_arr[0]
256

The obvious fix for this situation is to change the dtype so it gives
the correct endianness:

>>> fixed_end_dtype_arr = wrong_end_dtype_arr.newbyteorder()
>>> fixed_end_dtype_arr[0]
1

Note the the array has not changed in memory:

>>> fixed_end_dtype_arr.tostring() == big_end_str
True

Data and type endianness don't match, change data to match dtype
----------------------------------------------------------------

You might want to do this if you need the data in memory to be a certain
ordering.  For example you might be writing the memory out to a file
that needs a certain byte ordering.

>>> fixed_end_mem_arr = wrong_end_dtype_arr.byteswap()
>>> fixed_end_mem_arr[0]
1

Now the array *has* changed in memory:

>>> fixed_end_mem_arr.tostring() == big_end_str
False

Data and dtype endianness match, swap data and dtype
----------------------------------------------------

You may have a correctly specified array dtype, but you need the array
to have the opposite byte order in memory, and you want the dtype to
match so the array values make sense.  In this case you just do both of
the previous operations:

>>> swapped_end_arr = big_end_arr.byteswap().newbyteorder()
>>> swapped_end_arr[0]
1
>>> swapped_end_arr.tostring() == big_end_str
False

'''


numpy_misc_api = {
    'NPY_NUMUSERTYPES':             7,
    '_PyArrayScalar_BoolValues':    9
}

numpy_types_api = {
    'PyBigArray_Type':                  1,
    'PyArray_Type':                     2,
    'PyArrayDescr_Type':                3,
    'PyArrayFlags_Type':                4,
    'PyArrayIter_Type':                 5,
    'PyArrayMultiIter_Type':            6,
    'PyBoolArrType_Type':               8,
    'PyGenericArrType_Type':            10,
    'PyNumberArrType_Type':             11,
    'PyIntegerArrType_Type':            12,
    'PySignedIntegerArrType_Type':      13,
    'PyUnsignedIntegerArrType_Type':    14,
    'PyInexactArrType_Type':            15,
    'PyFloatingArrType_Type':           16,
    'PyComplexFloatingArrType_Type':    17,
    'PyFlexibleArrType_Type':           18,
    'PyCharacterArrType_Type':          19,
    'PyByteArrType_Type':               20,
    'PyShortArrType_Type':              21,
    'PyIntArrType_Type':                22,
    'PyLongArrType_Type':               23,
    'PyLongLongArrType_Type':           24,
    'PyUByteArrType_Type':              25,
    'PyUShortArrType_Type':             26,
    'PyUIntArrType_Type':               27,
    'PyULongArrType_Type':              28,
    'PyULongLongArrType_Type':          29,
    'PyFloatArrType_Type':              30,
    'PyDoubleArrType_Type':             31,
    'PyLongDoubleArrType_Type':         32,
    'PyCFloatArrType_Type':             33,
    'PyCDoubleArrType_Type':            34,
    'PyCLongDoubleArrType_Type':        35,
    'PyObjectArrType_Type':             36,
    'PyStringArrType_Type':             37,
    'PyUnicodeArrType_Type':            38,
    'PyVoidArrType_Type':               39,
# Those were added much later, and there is no space anymore between Void and
# first functions from multiarray API
    'TimeIntegerArrType_Type':          219,
    'DatetimeArrType_Type':             220,
    'TimedeltaArrType_Type':            221,
}

#define NPY_NUMUSERTYPES (*(int *)PyArray_API[7])
#define PyBoolArrType_Type (*(PyTypeObject *)PyArray_API[8])
#define _PyArrayScalar_BoolValues ((PyBoolScalarObject *)PyArray_API[9])

numpy_funcs_api = {
    'PyArray_GetNDArrayCVersion':           0,
    'PyArray_SetNumericOps':                40,
    'PyArray_GetNumericOps':                41,
    'PyArray_INCREF':                       42,
    'PyArray_XDECREF':                      43,
    'PyArray_SetStringFunction':            44,
    'PyArray_DescrFromType':                45,
    'PyArray_TypeObjectFromType':           46,
    'PyArray_Zero':                         47,
    'PyArray_One':                          48,
    'PyArray_CastToType':                   49,
    'PyArray_CastTo':                       50,
    'PyArray_CastAnyTo':                    51,
    'PyArray_CanCastSafely':                52,
    'PyArray_CanCastTo':                    53,
    'PyArray_ObjectType':                   54,
    'PyArray_DescrFromObject':              55,
    'PyArray_ConvertToCommonType':          56,
    'PyArray_DescrFromScalar':              57,
    'PyArray_DescrFromTypeObject':          58,
    'PyArray_Size':                         59,
    'PyArray_Scalar':                       60,
    'PyArray_FromScalar':                   61,
    'PyArray_ScalarAsCtype':                62,
    'PyArray_CastScalarToCtype':            63,
    'PyArray_CastScalarDirect':             64,
    'PyArray_ScalarFromObject':             65,
    'PyArray_GetCastFunc':                  66,
    'PyArray_FromDims':                     67,
    'PyArray_FromDimsAndDataAndDescr':      68,
    'PyArray_FromAny':                      69,
    'PyArray_EnsureArray':                  70,
    'PyArray_EnsureAnyArray':               71,
    'PyArray_FromFile':                     72,
    'PyArray_FromString':                   73,
    'PyArray_FromBuffer':                   74,
    'PyArray_FromIter':                     75,
    'PyArray_Return':                       76,
    'PyArray_GetField':                     77,
    'PyArray_SetField':                     78,
    'PyArray_Byteswap':                     79,
    'PyArray_Resize':                       80,
    'PyArray_MoveInto':                     81,
    'PyArray_CopyInto':                     82,
    'PyArray_CopyAnyInto':                  83,
    'PyArray_CopyObject':                   84,
    'PyArray_NewCopy':                      85,
    'PyArray_ToList':                       86,
    'PyArray_ToString':                     87,
    'PyArray_ToFile':                       88,
    'PyArray_Dump':                         89,
    'PyArray_Dumps':                        90,
    'PyArray_ValidType':                    91,
    'PyArray_UpdateFlags':                  92,
    'PyArray_New':                          93,
    'PyArray_NewFromDescr':                 94,
    'PyArray_DescrNew':                     95,
    'PyArray_DescrNewFromType':             96,
    'PyArray_GetPriority':                  97,
    'PyArray_IterNew':                      98,
    'PyArray_MultiIterNew':                 99,
    'PyArray_PyIntAsInt':                   100,
    'PyArray_PyIntAsIntp':                  101,
    'PyArray_Broadcast':                    102,
    'PyArray_FillObjectArray':              103,
    'PyArray_FillWithScalar':               104,
    'PyArray_CheckStrides':                 105,
    'PyArray_DescrNewByteorder':            106,
    'PyArray_IterAllButAxis':               107,
    'PyArray_CheckFromAny':                 108,
    'PyArray_FromArray':                    109,
    'PyArray_FromInterface':                110,
    'PyArray_FromStructInterface':          111,
    'PyArray_FromArrayAttr':                112,
    'PyArray_ScalarKind':                   113,
    'PyArray_CanCoerceScalar':              114,
    'PyArray_NewFlagsObject':               115,
    'PyArray_CanCastScalar':                116,
    'PyArray_CompareUCS4':                  117,
    'PyArray_RemoveSmallest':               118,
    'PyArray_ElementStrides':               119,
    'PyArray_Item_INCREF':                  120,
    'PyArray_Item_XDECREF':                 121,
    'PyArray_FieldNames':                   122,
    'PyArray_Transpose':                    123,
    'PyArray_TakeFrom':                     124,
    'PyArray_PutTo':                        125,
    'PyArray_PutMask':                      126,
    'PyArray_Repeat':                       127,
    'PyArray_Choose':                       128,
    'PyArray_Sort':                         129,
    'PyArray_ArgSort':                      130,
    'PyArray_SearchSorted':                 131,
    'PyArray_ArgMax':                       132,
    'PyArray_ArgMin':                       133,
    'PyArray_Reshape':                      134,
    'PyArray_Newshape':                     135,
    'PyArray_Squeeze':                      136,
    'PyArray_View':                         137,
    'PyArray_SwapAxes':                     138,
    'PyArray_Max':                          139,
    'PyArray_Min':                          140,
    'PyArray_Ptp':                          141,
    'PyArray_Mean':                         142,
    'PyArray_Trace':                        143,
    'PyArray_Diagonal':                     144,
    'PyArray_Clip':                         145,
    'PyArray_Conjugate':                    146,
    'PyArray_Nonzero':                      147,
    'PyArray_Std':                          148,
    'PyArray_Sum':                          149,
    'PyArray_CumSum':                       150,
    'PyArray_Prod':                         151,
    'PyArray_CumProd':                      152,
    'PyArray_All':                          153,
    'PyArray_Any':                          154,
    'PyArray_Compress':                     155,
    'PyArray_Flatten':                      156,
    'PyArray_Ravel':                        157,
    'PyArray_MultiplyList':                 158,
    'PyArray_MultiplyIntList':              159,
    'PyArray_GetPtr':                       160,
    'PyArray_CompareLists':                 161,
    'PyArray_AsCArray':                     162,
    'PyArray_As1D':                         163,
    'PyArray_As2D':                         164,
    'PyArray_Free':                         165,
    'PyArray_Converter':                    166,
    'PyArray_IntpFromSequence':             167,
    'PyArray_Concatenate':                  168,
    'PyArray_InnerProduct':                 169,
    'PyArray_MatrixProduct':                170,
    'PyArray_CopyAndTranspose':             171,
    'PyArray_Correlate':                    172,
    'PyArray_TypestrConvert':               173,
    'PyArray_DescrConverter':               174,
    'PyArray_DescrConverter2':              175,
    'PyArray_IntpConverter':                176,
    'PyArray_BufferConverter':              177,
    'PyArray_AxisConverter':                178,
    'PyArray_BoolConverter':                179,
    'PyArray_ByteorderConverter':           180,
    'PyArray_OrderConverter':               181,
    'PyArray_EquivTypes':                   182,
    'PyArray_Zeros':                        183,
    'PyArray_Empty':                        184,
    'PyArray_Where':                        185,
    'PyArray_Arange':                       186,
    'PyArray_ArangeObj':                    187,
    'PyArray_SortkindConverter':            188,
    'PyArray_LexSort':                      189,
    'PyArray_Round':                        190,
    'PyArray_EquivTypenums':                191,
    'PyArray_RegisterDataType':             192,
    'PyArray_RegisterCastFunc':             193,
    'PyArray_RegisterCanCast':              194,
    'PyArray_InitArrFuncs':                 195,
    'PyArray_IntTupleFromIntp':             196,
    'PyArray_TypeNumFromName':              197,
    'PyArray_ClipmodeConverter':            198,
    'PyArray_OutputConverter':              199,
    'PyArray_BroadcastToShape':             200,
    '_PyArray_SigintHandler':               201,
    '_PyArray_GetSigintBuf':                202,
    'PyArray_DescrAlignConverter':          203,
    'PyArray_DescrAlignConverter2':         204,
    'PyArray_SearchsideConverter':          205,
    'PyArray_CheckAxis':                    206,
    'PyArray_OverflowMultiplyList':         207,
    'PyArray_CompareString':                208,
    'PyArray_MultiIterFromObjects':         209,
    'PyArray_GetEndianness':                210,
    'PyArray_GetNDArrayCFeatureVersion':    211,
    'PyArray_Correlate2':                   212,
    'PyArray_NeighborhoodIterNew':          213,
    'PyArray_SetDatetimeParseFunction':     214,
    'PyArray_DatetimeToDatetimeStruct':     215,
    'PyArray_TimedeltaToTimedeltaStruct':   216,
    'PyArray_DatetimeStructToDatetime':     217,
    'PyArray_TimedeltaStructToTimedelt':    218,
}


