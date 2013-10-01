import pandas as pd
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt, pow, log, erf, abs, M_PI

ctypedef np.double_t DTYPE_t

@cython.cdivision(True)
@cython.boundscheck(False)
def univariate_kf(np.ndarray[double, ndim=1] raw, double seed, double q, double r):
    cdef:
        long nr = len(raw)
        double xhatprev,p=0,pprev=q,k=0,change,last_valid
        np.ndarray[DTYPE_t, ndim=1] res = np.zeros(nr, dtype=np.double) * np.NaN
    xhatprev = seed
    res[0] = seed
    for i from 1 <= i < nr:
        if not np.isnan(raw[i]):
            change = raw[i] - xhatprev
            res[i] = xhatprev + k*change
            p = (1-k)*pprev
            xhatprev = res[i]
            pprev = p + q
            k = pprev/(pprev+r)
        else:
            res[i] = xhatprev
        
    return res