import sys, os
# from matplotlib import rcParams, verbose

which = None, None

# First, see if --numarray or --Numeric was specified on the command
# line:
if hasattr(sys, 'argv'):        #Once again, Apache mod_python has no argv
    for a in sys.argv:
        if a in ["--Numeric", "--numeric", "--NUMERIC",
                 "--Numarray", "--numarray", "--NUMARRAY"]:
            which = a[2:], "command line"
            break
        del a

if os.getenv("NUMERIX"):
    which = os.getenv("NUMERIX"), "environment var"

# if which[0] is None:     
#    try:  # In theory, rcParams always has *some* value for numerix.
#        which = rcParams['numerix'], "rc"
#    except KeyError:
#        pass

# If all the above fail, default to Numeric.
if which[0] is None:
    which = "numeric", "defaulted"

which = which[0].strip().lower(), which[1]
if which[0] not in ["numeric", "numarray"]:
    verbose.report_error(__doc__)
    raise ValueError("numerix selector must be either 'Numeric' or 'numarray' but the value obtained from the %s was '%s'." % (which[1], which[0]))

if which[0] == "numarray":
