from utils import BlockMapper, add
from ls_utils import convert_rdd, add_index
import numpy as np
import logging
logger = logging.getLogger(__name__)

class RowMatrix(object):
    '''
    A row matrix class that holds the augmented matrix [A b]
    '''

    def __init__(self, rdd, name, m=None, n=None, cache=False, stack_type=1, repnum=1):
        '''
        rdd: RDD object that stores the matrix row wise,
        name: name of the matrix
        m, n: size of the matrix A
        cache: cache the matrix or not
        repnum: number of times to replicate the matrix vertically
        '''
        self.rdd_original = rdd
        self.stack_type = stack_type
        self.repnum = repnum

        if m is None:
            self.m_original, self.n = self.get_dimensions()
        else:
            self.m_original = m
            self.n = n

        self.rdd_original = convert_rdd(self.rdd_original)

        if repnum>1:
            self.name = name + '_stack' + str(stack_type) + '_rep' + str(self.repnum)
            if stack_type == 1:
                self.rdd = self.rdd_original.flatMap(lambda row:[row for i in range(repnum)])
                self.m = self.m_original*repnum
            elif stack_type == 2:
                n = self.n
                self.rdd = add_index(self.rdd_original).flatMap(lambda row: [row[0] for i in range(repnum)] if row[1]<self.m_original-n/2 else [row[0]])
                self.m = (self.m_original-self.n/2)*repnum + self.n/2
        else:
            self.name = name
            self.rdd = self.rdd_original
            self.m = m

        self.rdd_original = add_index(self.rdd_original)
        self.rdd = add_index(self.rdd)

        if cache:
            self.rdd.cache()
            #logger.info('number of rows: {0}'.format( self.rdd.count() )) # materialize the matrix

    def get_b(self):
        getb_mapper = GetbMapper()
        a = self.rdd.mapPartitions(getb_mapper).collect()

        b = np.zeros(self.m)
        for k, v in a:
            b[k] = v

        return b

    def rtimes_vec(self,vec):
        '''
        This code computes A*v
        '''
        # TO-DO: check dimension compatibility
        if vec.ndim > 1:
            vec = vec.squeeze()

        vec = self.rdd.context.broadcast(vec)

        matrix_rtimes_mapper = MatrixRtimesMapper()
        a = self.rdd.mapPartitions(lambda records: matrix_rtimes_mapper(records,vec=vec.value) ).collect()
        
        p = np.zeros(self.m)
        for k, v in a:
            p[k] = v

        return p

    def ltimes_vec(self,vec):
        '''
        This code computes u*A
        '''
        # TO-DO: check dimension compatibility
        if vec.ndim > 1:
            vec = vec_in_dict.squeeze()


        vec = self.rdd.context.broadcast(vec)

        matrix_ltimes_mapper = MatrixLtimesMapper()
        b = self.rdd.mapPartitions(lambda records: matrix_ltimes_mapper(records,vec=vec.value)).sum()

        return b

    def get_dimensions(self):
        m = self.matrix.count()
        try:
            n = len(self.matrix.first())
        except:
            n = 1
        return m, n
              
    def take(self, num_rows):
        return self.rdd.take(num_rows)

    def top(self):
        return self.rdd.first()

    def collect(self):
        return self.rdd.collect()

class GetbMapper(BlockMapper):

    def process(self):
        yield self.keys, np.vstack(self.data)[:,-1]

class MatrixRtimesMapper(BlockMapper):

    def process(self, vec):
        p = np.dot(np.vstack(self.data), np.append(vec,0)) # append the vector with zero since we are computing A*v
        yield self.keys, p

class MatrixLtimesMapper(BlockMapper):

    def __init__(self):
        BlockMapper.__init__(self)
        self.ba = None

    def process(self, vec):
        vec = vec[self.keys]

        if self.ba:
            self.ba += np.dot( vec, np.vstack(self.data) )
        else:
            self.ba = np.dot( vec, np.vstack(self.data) )

        return iter([])

    def close(self):
        yield self.ba[:-1]

