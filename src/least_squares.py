import numpy as np
import numpy.linalg as npl
from ls_utils import comp_l2_obj
from lsqr_spark import lsqr_spark
from comp_sketch import comp_sketch
import time
import logging
logger = logging.getLogger(__name__)

class RandLeastSquares:
    """
    This code implements randomized algorithms for least-squares regression problems on Spark.
    Author: Jiyan Yang (jiyan@stanford.edu)
    """
    def __init__(self, matrix_Ab, solver_type, **kwargs):
        """
        matrix_Ab: a RowMatrix object that holds the matrix [A b]
        solver_type: either 'high_precision' or 'low_precision'
        sketch_type: projection, sampling or None (for high-precision solver only)
        projection_type: cw, gaussian, rademacher or srdht
        c: projection size
        s: sampling size (for sampling sketch only)
        num_iters: number of iterations in LSQR (for high-precision solver only)
        k: number of independent trials to run
        """
        self.matrix_Ab = matrix_Ab
        self.solver_type = solver_type
        self.k = kwargs.get('k')
        self.params = kwargs

    def fit(self, load_N=True, save_N=False, debug=False):
        """
        load_N: load the precomputed N matrices if possible (it reduce the actual running time for sampling sketches)
        save_N: save the computed N matrices for future use
        """

        if self.solver_type == 'low_precision':
            logger.info('Ready to start computing solutions!')
            x, time = comp_sketch(self.matrix_Ab, 'x', load_N, save_N, **self.params)

        elif self.solver_type == 'high_precision':
            num_iters = self.params.get('num_iters')
            sketch_type = self.params.get('sketch_type')

            # start computing a sketch
            if sketch_type is not None:
                N, time_proj = comp_sketch(self.matrix_Ab, 'N', load_N, save_N, **self.params)
            else:
                N = [np.eye(self.matrix_Ab.n-1)]
                self.k = 1
                time_proj = 0

            b = []

            # start lsqr
            time = [time_proj for i in range(num_iters)]
            x = []
 
            for i in range(self.k):
                x_iter, y_iter, time_iter = lsqr_spark(self.matrix_Ab,b,self.matrix_Ab.m,self.matrix_Ab.n-1,N[i],1e-10,num_iters)
                x.append(x_iter)
                time = [time[i] + time_iter[i] for i in range(num_iters)]
            
        else:
            raise ValueError("invalid solver_type")

        self.x = x
        self.time = time

        if debug:
            print self.x

    def __comp_cost(self,x,stack_type,repnum,b=None):
        if stack_type == 1:
            costs = [ comp_l2_obj(self.matrix_Ab.rdd_original,np.array(p)) for p in x ]
        elif stack_type == 2: 
            n = self.matrix_Ab.n
            a = [ repnum*comp_l2_obj(self.matrix_Ab.matrix_original,p)**2 - (repnum-1)*npl.norm( p[n/2:] - b[-n/2:])**2 for p in x ]
            costs = [ np.sqrt(aa)/np.sqrt(repnum) for aa in a ]

        return costs

    def comp_relerr(self,x_opt,f_opt,b=None):
        """
        Evaluating the accuracy of solutions
            b: response vector
            x_opt: optimal solution
            f_opt: optimal objetive value
        """

        logger.info('Evaluating solution qualities!')
        x_relerr = []
        f_relerr = []

        if self.solver_type == 'high_precision':
            costs = [self.__comp_cost(x,self.matrix_Ab.stack_type,self.matrix_Ab.repnum,b) for x in self.x]
            f_relerr = [ (np.abs(f_opt- np.array(c))/f_opt).tolist() for c in costs ]
            x_relerr = [ [ npl.norm( p - x_opt ) / npl.norm(x_opt) for p in self.x[i] ] for i in range(self.k) ] # a list of list. each element is a list with length self.iter

            f_final = [ p[-1] for p in f_relerr ]
            idx = np.where(f_final == np.median(f_final))[0]
            if len(idx) > 1:
                idx = idx[0]
            self.x_relerr_median = x_relerr[idx]
            self.f_relerr_median = f_relerr[idx]
        else:
            costs = self.__comp_cost(self.x,self.matrix_Ab.stack_type,self.matrix_Ab.repnum,b)
            f_relerr = [ np.abs(f_opt - c)/f_opt for c in costs ]
            x_relerr = [ npl.norm(p - x_opt)/npl.norm(x_opt) for p in self.x ]

            self.x_relerr_median = np.median(x_relerr)
            self.f_relerr_median = np.median(f_relerr)

        return self.x_relerr_median, self.f_relerr_median

