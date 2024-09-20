from numpy import deprecate
refft = deprecate(rfft, 'refft', 'rfft')
irefft = deprecate(irfft, 'irefft', 'irfft')
refft2 = deprecate(rfft2, 'refft2', 'rfft2')
irefft2 = deprecate(irfft2, 'irefft2', 'irfft2')
refftn = deprecate(rfftn, 'refftn', 'rfftn')
irefftn = deprecate(irfftn, 'irefftn', 'irfftn')



__all__ = ['average', 'histogram', 'standardDeviation', 'variance']

