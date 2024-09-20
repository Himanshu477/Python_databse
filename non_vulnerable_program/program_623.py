import Numeric
from RNG import *

standard_generator = CreateGenerator(-1)

def ranf():
        "ranf() = a random number from the standard generator."
        return standard_generator.ranf()

def random_sample(*n):
        """random_sample(n) = array of n random numbers;
        random_sample(n1, n2, ...)= random array of shape (n1, n2, ..)"""

        if not n:
                return standard_generator.sample(1)
        m = 1
        for i in n:
                m = m * i
        return Numeric.reshape (standard_generator.sample(m), n)



