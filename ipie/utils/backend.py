try:
    import cupy
    _have_cupy = True
except ImportError:
    _have_cupy = False

import numpy

from ipie.config import IPIE_USE_GPU

if IPIE_USE_GPU and _have_cupy:
    numlib = cupy
    numlib.to_host = cupy.asnumpy
else:
    numlib = numpy
    numlib.to_host = numpy.array
