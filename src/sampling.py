import numpy as np
from utils import add, BlockMapper
from ls_utils import get_x, get_N
import logging
logger = logging.getLogger(__name__)

class Sampling(object):
    """
    Given matrix A, the class computes performs random sampling on A and performs further operations on SA.
    It returns the solution 'x' to the subproblem induced by PA or the matrix 'N' such that SA[:,:-1]*inv(N) is a matrix with orthonormal columns.
    """

    def __init__(self, N):
        """
        N: precomputed N matrices from projections
        """
        self.N = N

    def execute(self, matrix, objective, s, return_N=False):
        """
        matrix: a RowMatrix object storing the matrix A
        objective: either 'x' or 'N'
            'x': the function returns the solution to the problem min_x || SA[:,:-1]x - SA[:,-1] ||_2
            'N': the function returns a square matrix N such that SA[:,:-1]*inv(N) is a matrix with orthonormal columns
        s: samping size
        return_N: when the objective is 'x', whether to return the matrix N which makes SA[:,:-1]*inv(N) has orthonormal columns
        """
        logger.info('In sampling, computing {0}!'.format(objective))
        lev_sum = self.__get_lev_sum(matrix)
        SA = self.__sample(matrix, s, lev_sum)

        if objective == 'x':
            return SA.map(lambda (key,sa): get_x(sa,return_N)).collect()
        elif objective == 'N':
            return SA.map(lambda (key,sa): get_N(sa)).collect()
        else:
            raise ValueError('Please enter a valid objective!')

    def __get_lev_sum(self, matrix):
        N = matrix.rdd.context.broadcast(self.N)
        lsm = GetLevSumMapper()
        return matrix.rdd.mapPartitions(lambda records: lsm(records, N=N.value)).sum()

    def __sample(self, matrix, s, lev_sum):
        N = matrix.rdd.context.broadcast(self.N)
        sm = SampleMapper()
        return matrix.rdd.mapPartitions(lambda records: sm(records, N=N.value, s=s, lev_sum=lev_sum)).reduceByKey(lambda x,y: np.vstack((x,y)))

class GetLevSumMapper(BlockMapper):
    def process(self, N):
        data = np.vstack(self.data)[:,:-1]
        lev_sum = []
        for N1 in N:
            lev_sum.append( np.sum( np.sum( np.dot(data, N1)**2, axis=1)) )
        yield np.array(lev_sum)

class SampleMapper(BlockMapper):
    def process(self, N, s, lev_sum):
        data = np.vstack(self.data)
        np.random.seed()
        for i in xrange(len(N)):
            lev = np.sum( np.dot(data[:,:-1], N[i])**2, axis=1)
            p = lev*s/lev_sum[i]
            p[ p>1 ] = 1.0
            sampled_idx = np.random.rand(len(p)) < p
            yield (i, (1./p[sampled_idx])[:,None]*data[sampled_idx,])


