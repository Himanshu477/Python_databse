import numpy
result = numpy.test(options.mode,
                    verbose=options.verbose,
                    extra_argv=args,
                    doctests=options.doctests,
                    coverage=options.coverage)

if result.wasSuccessful():
    sys.exit(0)
else:
    sys.exit(1)


