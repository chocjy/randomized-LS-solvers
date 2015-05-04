from math import sqrt, log

import numpy as np
from numpy.linalg import norm, lstsq

import time
import logging
logger = logging.getLogger(__name__)

def lsqr_spark( matrix_Ab, b, m, n, N, tol=1e-14, iter_lim=None):
    """
    A simple version of LSQR on Spark
    """

    x_iter = []
    time_iter = []
    t0 = time.time()

    logger.info('In LSQR!')
    eps  = 32*np.finfo(float).eps;      # slightly larger than eps

    if tol < eps:
        tol = eps
    elif tol >= 1:
        tol = 1-eps

    max_n_stag = 3

    # getting b (TO-DO: make it faster)
    #vec = np.zeros((n+1,1))
    #vec[n,0] = 1 
    #u = matrix_Ab.rtimes(vec).squeeze()

    u = matrix_Ab.get_b() # u is a dict

    beta = norm(np.array(u.values)) 
    #u   /= beta

    v = np.dot( matrix_Ab.ltimes_vec(u), N ).squeeze() # v is an array
    v /= beta

    alpha = norm(v)
    if alpha != 0:
       v    /= alpha

    w     = v.copy()
    x     = np.zeros(n)

    phibar = beta
    rhobar = alpha

    nrm_a    = 0.0
    cnd_a    = 0.0
    sq_d     = 0.0
    nrm_r    = beta
    nrm_ar_0 = alpha*beta

    if nrm_ar_0 == 0:                     # alpha == 0 || beta == 0
        return x, 0, 0

    nrm_x  = 0
    sq_x   = 0
    z      = 0
    cs2    = -1
    sn2    = 0

    stag   = 0

    flag = -1
    if iter_lim is None:
        iter_lim = np.max( [20, 2*np.min([m,n])] )

    for itn in xrange(int(iter_lim)):

        p = matrix_Ab.rtimes_vec(np.dot(N,v))
        temp = dict()
        elem_sq_sum = 0
        for k in p:
            temp[k] = p[k] - alpha*u[k]
            elem_sq_sum += temp[k]**2
        u = temp
        beta = np.sqrt(elem_sq_sum)

        #u = matrix_Ab.rtimes_vec(np.dot(N,v)).squeeze() - alpha*u
        #beta = norm(u)
        #u   /= beta

        nrm_a = sqrt(nrm_a**2 + alpha**2 + beta**2)

        v = np.dot( matrix_Ab.ltimes_vec(u), N).squeeze()/beta - beta*v

        alpha = norm(v)
        v    /= alpha

        rho    =  sqrt(rhobar**2+beta**2)
        cs     =  rhobar/rho
        sn     =  beta/rho
        theta  =  sn*alpha
        rhobar = -cs*alpha
        phi    =  cs*phibar
        phibar =  sn*phibar

        x      = x + (phi/rho)*w
        w      = v-(theta/rho)*w

        # estimate of norm(r)
        nrm_r   = phibar

        # estimate of norm(A'*r)
        nrm_ar  = phibar*alpha*np.abs(cs)

        # check convergence
        if nrm_ar < tol*nrm_ar_0:
            flag = 0
        #    break

        if nrm_ar < eps*nrm_a*nrm_r:
            flag = 0
        #    break

        # estimate of cond(A)
        sq_w    = np.dot(w,w)
        nrm_w   = sqrt(sq_w)
        sq_d   += sq_w/(rho**2)
        cnd_a   = nrm_a*sqrt(sq_d)

        # check condition number
        if cnd_a > 1/eps:
            flag = 1
        #    break

        # check stagnation
        if abs(phi/rho)*nrm_w < eps*nrm_x:
            stag += 1
        else:
            stag  = 0
        if stag >= max_n_stag:
            flag = 1
        #    break

        # estimate of norm(x)
        delta   =  sn2*rho
        gambar  = -cs2*rho
        rhs     =  phi - delta*z
        zbar    =  rhs/gambar
        nrm_x   =  sqrt(sq_x + zbar**2)
        gamma   =  sqrt(gambar**2 + theta**2)
        cs2     =  gambar/gamma
        sn2     =  theta /gamma
        z       =  rhs   /gamma
        sq_x   +=  z**2

        x_iter.append(x)
        time_iter.append( time.time() - t0 )

        logger.info("Finished one iteration!")

    y_iter = x_iter
    x_iter = [np.dot(N,x) for x in x_iter]
    return x_iter, y_iter, time_iter
        
    
