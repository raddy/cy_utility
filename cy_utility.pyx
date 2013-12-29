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

@cython.cdivision(True)
@cython.boundscheck(False)
def assymetrical_univariate_kf(np.ndarray[double, ndim=1] raw, double seed, double q, double r, double ratio=.5):
    cdef:
        long nr = len(raw)
        double xhatprev,p=0,pprev=q,k=0,k2=0,change,last_valid
        np.ndarray[DTYPE_t, ndim=2] res = np.zeros([nr,2], dtype=np.double) * np.NaN
    xhatprev = seed
    res[0][0] = seed
    res[0][1] = seed
    for i from 1 <= i < nr:
        if not np.isnan(raw[i]):
            change = raw[i] - xhatprev
            if change>0:
                res[i][0] = xhatprev + k*change
                res[i][1] = xhatprev + k2*change
                xhatprev = res[i][0]
            else:
                res[i][0] = xhatprev + k2*change
                res[i][1] = xhatprev + k*change
                xhatprev = res[i][1]
            p = (1-k)*pprev
            pprev = p + q
            k = pprev/(pprev+r)
            k2 = pprev/(pprev+r*ratio)
        else:
            res[i][0] = xhatprev
            res[i][1] = xhatprev
        
    return res    

@cython.cdivision(True)
@cython.boundscheck(False)
def find_turns(np.ndarray[long,ndim=1] bids,np.ndarray[long,ndim=1] asks):
    cdef:
        long blen = bids.shape[0]
        np.ndarray[long, ndim=1] res = np.zeros(blen, dtype=np.int64)
        long prev_bid,prev_ask
    prev_bid = bids[0]
    prev_ask = asks[0]
    for i from 1<= i < blen:
        if not (bids[i]>0 and asks[i]>0):
            continue
        if bids[i]>=prev_ask:
            res[i] = 1
        elif asks[i]<=prev_bid:
            res[i] = -1
        else:
            continue
        prev_bid = bids[i]
        prev_ask = asks[i]   
    return res