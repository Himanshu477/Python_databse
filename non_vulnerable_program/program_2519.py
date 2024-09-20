from optparse import OptionParser
parser = OptionParser("usage: %prog [options] -- [nosetests options]")
parser.add_option("-v", "--verbose",
                  action="count", dest="verbose", default=1,
                  help="increase verbosity")
parser.add_option("--doctests",
                  action="store_true", dest="doctests", default=False,
                  help="Run doctests in module")
parser.add_option("--coverage",
                  action="store_true", dest="coverage", default=False,
                  help="report coverage of NumPy code (requires 'coverage' module")
parser.add_option("-m", "--mode",
                  action="store", dest="mode", default="fast",
                  help="'fast', 'full', or something that could be "
                       "passed to nosetests -A [default: %default]")
(options, args) = parser.parse_args()

