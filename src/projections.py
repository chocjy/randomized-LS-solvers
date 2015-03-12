from utils import add, BlockMapper
from ls_utils import get_x, get_N
import numpy.linalg as npl
import numpy as np
import logging
logger = logging.getLogger(__name__)

class Projections(object):
    """
    Given matrix A, the class computes performs random projections on A and performs further operations on PA.
    It returns the solution 'x' to the subproblem induced by PA or the matrix 'N' such that PA[:,:-1]*inv(N) is a matrix with orthonormal columns.
    """

    def __init__(self, **kwargs):
        """
        projection_type: cw, gaussian, rademacher or srdht
        c: projection size
        k: number of independent trials to run
        """
        self.projection_type = kwargs.pop('projection_type', 'cw')
        self.k = kwargs.pop('k',1)
        self.c = kwargs.pop('c')
        self.__validate()

    def __validate(self):
        if self.projection_type not in Projections.projection_type:
            raise NotImplementedError('%s projection_type not yet implemented' % self.projection_type)
        if not self.c:
            raise ValueError('"c" param is missing')

    def execute(self, rowmatrix, objective, return_N=False):
        """
        matrix: a RowMatrix object storing the matrix A
        objective: either 'x' or 'N'
            'x': the function returns the solution to the problem min_x || PA[:,:-1]x - PA[:,-1] ||_2
            'N': the function returns a square matrix N such that PA[:,:-1]*inv(N) is a matrix with orthonormal columns
        return_N: when the objective is 'x', whether to return the matrix N which makes PA[:,:-1]*inv(N) has orthonormal columns
        """
        logger.info('In projections, computing {0}!'.format(objective))
        PA = self.__project(rowmatrix)
        if objective == 'x':
            return PA.map(lambda (key,pa): get_x(pa,return_N)).collect()
        elif objective == 'N':
            return PA.map(lambda (key,pa): get_N(pa)).collect()
        else:
            raise ValueError('Please enter a valid objective!')

    def __project(self, rowmatrix):
        c = self.c
        k = self.k
        if self.projection_type == 'cw':
            cwm = CWMapper(self.c, self.k)
            PA = rowmatrix.rdd.mapPartitions(cwm)
        elif self.projection_type == 'gaussian':
            gm = GaussianMapper(self.c, self.k)
            PA = rowmatrix.rdd.mapPartitions(gm)
        elif self.projection_type == 'rademacher':
            rm = RademacherMapper(self.c, self.k)
            PA = rowmatrix.rdd.mapPartitions(rm)
        elif self.projection_type == 'srdht':
            seed_s = np.random.randint(100000,size=k)
            srdm = SRDHTMapper(self.c, self.k, rowmatrix.m, seed_s)
            PA = rowmatrix.rdd.mapPartitions(srdm)

        PA = PA.reduceByKey(add).map(lambda (key,pa): (key[0],pa)).reduceByKey(lambda x,y: np.vstack((x,y)))

        return PA

    projection_type = ['cw','gaussian','rademacher','srdht']


class CWMapper(BlockMapper):
    def __init__(self,c,k):
        BlockMapper.__init__(self,1)
        self.c = c
        self.k = k

    def process(self):
        row = self.data[0]
        np.random.seed()
        rt = np.random.randint(self.c,size=self.k).tolist()
        coin = (np.random.rand(self.k)<0.5)*2-1
        for i in xrange(self.k):
            yield ((i,rt[i]),coin[i]*row)

class GaussianMapper(BlockMapper):
    def __init__(self,c,k):
        BlockMapper.__init__(self,1000)
        self.c = c  # projection size
        self.k = k  # number of independent trials
        self.PA = [None for i in range(self.k)]

    def process(self):
        data = np.vstack(self.data)
        r = data.shape[0]

        np.random.seed()
        for i in xrange(self.k):
            if self.PA[i] is None:
                self.PA[i] = np.dot(np.random.randn(self.c,r),data)/np.sqrt(self.c)
            else:
                self.PA[i] += np.dot(np.random.randn(self.c,r),data)/np.sqrt(self.c)

        return iter([])

    def close(self):
        for i in xrange(self.k):
            block_sz = 500
            m = self.PA[0].shape[0]
            start_idx = np.arange(0, m, block_sz)
            end_idx = np.append(np.arange(block_sz, m, block_sz), m)

            for j in range(len(start_idx)):
                yield (i,j), self.PA[i][start_idx[j]:end_idx[j],:]

class RademacherMapper(BlockMapper):
    def __init__(self,c,k):
        BlockMapper.__init__(self,1000)
        self.c = c
        self.k = k
        self.PA = [None for i in range(self.k)]

    def process(self):
        data = np.vstack(self.data)
        r = data.shape[0]

        np.random.seed()
        for i in xrange(self.k):
            if self.PA[i] is None:
                self.PA[i] = np.dot((np.random.rand(self.c,r)<0.5)*2-1,data)/np.sqrt(self.c)
            else:
                self.PA[i] += np.dot((np.random.rand(self.c,r)<0.5)*2-1,data)/np.sqrt(self.c)
   
        return iter([])

    def close(self):
        for i in xrange(self.k):
            block_sz = 500
            m = self.PA[0].shape[0]
            start_idx = np.arange(0, m, block_sz)
            end_idx = np.append(np.arange(block_sz, m, block_sz), m)

            for j in range(len(start_idx)):
                yield (i,j), self.PA[i][start_idx[j]:end_idx[j],:]

class SRDHTMapper(BlockMapper):
    def __init__(self,c,k,m,seed_s):
        BlockMapper.__init__(self)
        self.m = m
        self.seed_s = seed_s
        self.c = c
        self.k = k
        self.PA = [None for i in range(self.k)]

    def process(self):
        data = np.vstack(self.data)
        r = data.shape[0]
        row_idx = np.array(self.keys)

        for i in xrange(self.k):
            S = np.arange(self.m)
            np.random.seed(self.seed_s[i])
            np.random.shuffle(S)
            S = S[:self.c]
            np.random.seed()
            rs = (np.random.rand(r)<0.5)*2-1
            rand_data = np.dot(np.diag(rs),data)

            if self.PA[i] is None:
                self.PA[i] = np.dot( np.sqrt(2)*np.cos(2*np.pi*np.outer(S,row_idx)/self.m-np.pi/4), rand_data)/np.sqrt(self.m)
            else:
                self.PA[i] += np.dot( np.sqrt(2)*np.cos(2*np.pi*np.outer(S,row_idx)/self.m-np.pi/4), rand_data)/np.sqrt(self.m) 

        return iter([])

    def close(self):
        for i in xrange(self.k):
            block_sz = 500
            m = self.PA[0].shape[0]
            start_idx = np.arange(0, m, block_sz)
            end_idx = np.append(np.arange(block_sz, m, block_sz), m)

            for j in range(len(start_idx)):
                yield (i,j), self.PA[i][start_idx[j]:end_idx[j],:]


