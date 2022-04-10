try:
    import cupy
    _have_cupy = True
except ImportError:
    _have_cupy = False

import numpy

from ipie.config import config

_use_gpu = config.get_option('use_gpu')

_numlib_c = cupy
_numlib_c.to_host = cupy.asnumpy
_numlib_n = numpy
_numlib_n.to_host = numpy.array

if _use_gpu and _have_cupy:
    numlib = _numlib_c
else:
    numlib = _numlib_n

print(numlib)
print("config: ", config)
