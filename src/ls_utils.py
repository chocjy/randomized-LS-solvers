import numpy.linalg as npl
import numpy as np

from utils import *

def convert_rdd(rdd):
    row = rdd.first()
    if isinstance(row, unicode):
        rdd = rdd.map(lambda row: np.array([float(x) for x in row.split(' ')]))
    else:
        rdd = rdd.map(lambda row: np.array(row))

    return rdd

def comp_l2_obj(Ab_rdd, x):
    # x is a np array
    return np.sqrt( Ab_rdd.map( lambda (key,row): (np.dot(row[:-1],x) - row[-1])**2 ).reduce(add) )

def add_index(rdd): 
    starts = [0] 
    nums = rdd.mapPartitions(lambda it: [sum(1 for i in it)]).collect() 
    for i in range(len(nums) - 1): 
        starts.append(starts[-1] + nums[i]) 

    def func(k, it): 
        for i, v in enumerate(it, starts[k]): 
            yield i, v

    return rdd.mapPartitionsWithIndex(func)

def get_x(pa,return_N=False):
    A = pa[:,:-1]
    b = pa[:,-1]
    m = A.shape[0]

    [U, s, V] = npl.svd(A, 0)
    N = V.transpose()/s

    if return_N:
        return (N, np.dot(N, np.dot(U.T,b)))
    else:
        return np.dot(N, np.dot(U.T,b))

def get_N(pa):
    [U, s, V] = npl.svd(pa[:,:-1], 0)
    N = V.transpose()/s
    return N
