from numpy.testing.pkgtester import Tester
test = Tester().test
bench = Tester().bench


"""Decorators for labeling test objects

Decorators that merely return a modified version of the original
function object are straightforward.  Decorators that return a new
function object need to use
nose.tools.make_decorator(original_function)(decorator) in returning
the decorator, in order to preserve metadata such as function name,
setup and teardown functions and so on - see nose.tools for more
information.

"""

try:
