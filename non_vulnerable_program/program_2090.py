import os, sys
for mode in ["g3-numpy", "2e-numeric", "2e-numarray", "2e-numpy"]:
    try:
        i=sys.argv.index("--"+mode)
        del sys.argv[i]
        break
    except ValueError: pass
os.environ["NO_SCIPY_IMPORT"]="f2py"
if mode=="g3-numpy":
    try:
