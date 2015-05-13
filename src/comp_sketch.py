from utils import pickle_write, pickle_load
from projections import Projections
from sampling import Sampling
import time
import os
import logging
logger = logging.getLogger(__name__)

def comp_sketch(matrix, objective, load_N=False, save_N=False, N_dir='../N_file/', **kwargs):
    """
    Given matrix A, the function comp_sketch computes a sketch for A and performs further operations on PA.
    It returns the total running time and the desired quantity.

    parameter:
        matrix: a RowMatrix object storing the matrix [A b]
        objective: either 'x' or 'N'
            'x': the function returns the solution to the problem min_x || PA[:,:-1]x - PA[:,-1] ||_2
            'N': the function returns a square matrix N such that PA[:,:-1]*inv(N) is a matrix with orthonormal columns
        load_N: load the precomputed N matrices if possible (it reduces the actual running time for sampling sketches)
        save_N: save the computed N matrices for future use
        sketch_type: either 'projection' or 'sampling'
        projection_type: cw, gaussian, rademacher or srdht
        c: projection size
        s: sampling size (for sampling sketch only)
        k: number of independent trials to run
    """

    sketch_type = kwargs.get('sketch_type')

    if not os.path.exists(N_dir):
        os.makedirs(N_dir)

    if objective == 'x':
        
        if sketch_type == 'projection':
            projection = Projections(**kwargs)
            t = time.time()
            x = projection.execute(matrix, 'x', save_N)
            t = time.time() - t

            if save_N:
                logger.info('Saving N matrices from projections!')
                N = [a[0] for a in x]
                x = [a[1] for a in x]
                # saving N
                filename = N_dir + 'N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k')))+ '.dat'
                data = {'N': N, 'time': t}
                pickle_write(filename,data)
 
        elif sketch_type == 'sampling':
            s = kwargs.get('s')
            new_N_proj = 0
            N_proj_filename = N_dir + 'N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) +'.dat'

            if load_N and os.path.isfile(N_proj_filename):
                logger.info('Found N matrices from projections, loading them!')
                N_proj_filename = N_dir + 'N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) +'.dat'
                result = pickle_load(N_proj_filename)
                N_proj = result['N']
                t_proj = result['time']
            else: # otherwise, compute it
                t = time.time()
                projection = Projections(**kwargs)
                N_proj = projection.execute(matrix, 'N')
                t_proj = time.time() - t
                new_N_proj = 1

            sampling = Sampling(N=N_proj)
            t = time.time()
            x = sampling.execute(matrix, 'x', s, save_N )
            t = time.time() - t + t_proj

            if save_N and new_N_proj:
                logger.info('Saving N matrices from projections!')
                filename = N_dir + 'N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) + '.dat'
                data = {'N': N_proj, 'time': t_proj}
                pickle_write(filename,data)

            if save_N:
                logger.info('Saving N matrices from sampling!')
                N = [a[0] for a in x]
                x = [a[1] for a in x]
                filename = N_dir + 'N_' + matrix.name + '_sampling_s' + str(int(kwargs.get('s'))) + '_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) + '.dat'
                data = {'N': N, 'time': t}
                pickle_write(filename,data)

        else:
            raise ValueError('Please enter a valid sketch type!')
        return x, t

    elif objective == 'N':
        if sketch_type == 'projection':
            N_proj_filename = N_dir + 'N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) + '.dat'

            if load_N and os.path.isfile(N_proj_filename):
                logger.info('Found N matrices from projections, loading them!')
                result = pickle_load(filename)
                N = result['N']
                t = result['time']
            else:
                t = time.time()
                projection = Projections(**kwargs)
                N = projection.execute(matrix, 'N')
                t = time.time() - t

                if save_N:
                    logger.info('Saving N matrices from projections!')
                    filename = N_dir + 'N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k')))+ '.dat'
                    data = {'N': N, 'time': t}
                    pickle_write(filename,data)

        elif sketch_type == 'sampling':
            s = kwargs.get('s')
            new_N_proj = 0
            new_N_samp = 0

            N_samp_filename = N_dir + 'N_' + matrix.name + '_sampling_s' + str(int(kwargs.get('s'))) + '_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) + '.dat'
            N_proj_filename = N_dir + 'N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) + '.dat'

            if load_N and os.path.isfile(N_samp_filename):
                logger.info('Found N matrices from sampling, loading them!')
                result = pickle_load(N_samp_filename)
                N = result['N']
                t = result['time']

            elif load_N and os.path.isfile(N_proj_filename):
                logger.info('Found N matrices from projections, loading them!')
                result = pickle_load(N_proj_filename)
                N_proj = result['N']
                t_proj = result['time']

                sampling = Sampling(N=N_proj)
                t = time.time()
                N = sampling.execute(matrix, 'N', s)
                t = time.time() - t + t_proj
                new_N_samp = 1

            else:
                t = time.time()
                projection = Projections(**kwargs)
                N_proj = projection.execute(matrix, 'N')
                t_proj = time.time() - t
                new_N_proj = 1

                t = time.time()
                sampling = Sampling(N=N_proj)
                N = sampling.execute(matrix, 'N', s)
                t = time.time() - t + t_proj
                new_N_samp = 1

            if save_N and new_N_proj:
                logger.info('Saving N matrices from projections!')
                filename = N_dir + 'N_' + matrix.name + '_projection_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) + '.dat'
                data = {'N': N_proj, 'time': t_proj}
                pickle_write(filename,data)

            if save_N and new_N_samp:
                logger.info('Saving N matrices from sampling!')
                filename = N_dir + 'N_' + matrix.name + '_sampling_s' + str(int(kwargs.get('s'))) + '_' + kwargs.get('projection_type') + '_c' + str(int(kwargs.get('c'))) + '_k' + str(int(kwargs.get('k'))) + '.dat'
                data = {'N': N_proj, 'time': t}
                pickle_write(filename,data)

        else:
            raise ValueError('Please enter a valid sketch type!')
        return N, t
    else:
        raise ValueError('Please enter a valid objective!')

if __name__ == '__main__':
    _test()

