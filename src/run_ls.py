from pyspark import SparkContext
from pyspark import SparkConf
from least_squares import RandLeastSquares
from rowmatrix import RowMatrix
from utils import pickle_write
import time
import sys
import os
import argparse
import ConfigParser
import scipy.stats
import numpy as np
import logging.config

def print_params(args, logger):
    logger.info('------------------------------------------------------------')
    logger.info('---------------Solving Least Squares Problems---------------')
    logger.info('------------------------------------------------------------')
    logger.info('dataset: {0}'.format( args.dataset ))
    logger.info('size: {0} by {1}'.format( args.dims[0], args.dims[1] ))
    logger.info('loading file from {0}'.format( args.file_source ))
    if args.nrepetitions>1:
        logger.info('number of repetitions: {0}'.format( args.nrepetitions ))
        logger.info('stack type is: {0}'.format( args.stack_type ))
    logger.info('number of partitions: {0}'.format( args.npartitions ))
    if args.cache:
        logger.info('cache the dataset')
    logger.info('Results will be stored to {0}'.format(args.output_filename))
    logger.info('------------------------------------------------------------')
    logger.info('loading setting configuration file from {0}'.format(args.setting_filename))
    logger.info('loading logging configuration file from {0}'.format(args.logging_filename))
    logger.info('------------------------------------------------------------')
    logger.info('solver: {0}'.format( args.solver_type ))
    logger.info('sketch type: {0}'.format( args.sketch_type ))
    if args.sketch_type:
        logger.info('projection type: {0}'.format( args.projection_type ))
        logger.info('projection size: {0}'.format( args.r ))
        if args.sketch_type == 'sampling':
            logger.info('sampling size: {0}'.format( args.s ))
    if args.solver_type == 'high_precision':
        logger.info('number of iterations: {0}'.format( args.q ))
    logger.info('number of independent trial: {0}'.format( args.k ))
    logger.info('------------------------------------------------------------')
    if args.load_N:
        logger.info('Will load N matrices wheneve it is possible!')
    if args.save_N:
        logger.info('Will save N matrices wheneve it is possible!')
    if args.test:
        logger.info('Will test the returned solutions!')
    if args.save_logs:
        logger.info('Logs will be saved!')
    logger.info('------------------------------------------------------------')    
    if args.debug:
        logger.info('Debug mode is on!')
        logger.info('------------------------------------------------------------')

class ArgumentError(Exception):
    pass

class OptionError(Exception):
    pass

def main(argv):
    parser = argparse.ArgumentParser(description='Getting parameters.',prog='run_ls.sh')

    parser.add_argument('dataset', type=str, help='dataset_Ab.txt stores the input matrix to run CX on; \
           dataset.txt stores the original matrix (only needed for -t);')
    parser.add_argument('--dims', metavar=('m','n'), type=int, nargs=2, required=True, help='size of the input matrix')
    parser.add_argument('--nrepetitions', metavar='numRepetitions', default=1, type=int, help='number of times to stack matrix vertically in order to generate large matrices')
    parser.add_argument('--stack', metavar='stackType', dest='stack_type', type=int, default=1, help='stack type')
    parser.add_argument('--npartitions', metavar='numPartitions', default=280, type=int, help='number of partitions in Spark')
    parser.add_argument('--setting_filename', metavar='settingConfFilename', default='conf/settings.cfg', type=str, help='name of the configuration file storing the settings')
    parser.add_argument('--logging_filename', metavar='loggingConfFilename', default='conf/logging.cfg', type=str, help='configuration file for Python logging')
    parser.add_argument('-c', '--cache', action='store_true', help='cache the dataset in Spark')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--hdfs', dest='file_source', default='local', action='store_const', const='hdfs', help='load dataset from HDFS (default: loading files from local)')
    group.add_argument('--s3', dest='file_source', default='local', action='store_const', const='s3', help='load dataset from Amazon S3 (default: loading files from local)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--low-precision', dest='solver_type', default='low_precision', action='store_const', const='low_precision', help='use low-precision solver')
    group.add_argument('--high_precision', dest='solver_type', default='low_precision', action='store_const', const='high_precision', help='use high_precision solver')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--projection', dest='sketch_type', action='store_const', const='projection', help='compute sketch by projection')
    group.add_argument('--sampling', dest='sketch_type', action='store_const', const='sampling', help='compute sketch by sampling')
    parser.add_argument('-p', dest='projection_type', default='gaussian', choices=('cw','gaussian','rademacher','srdht'), help='underlying projection type')
    parser.add_argument('-r', metavar='projectionSize', type=int, help='sketch size')
    parser.add_argument('-s', metavar='samplingSize', type=int, help='sampling size (for samping sektch only)')
    parser.add_argument('-q', '--niters', metavar='numIters', dest='q', type=int, help='number of iterations in LSQR')
    parser.add_argument('-k', '--ntrials', metavar='numTrials', dest='k', default=1, type=int, help='number of independent trials to run')
    parser.add_argument('-t', '--test', action='store_true', help='compute accuracies of the returned solutions')
    parser.add_argument('--save_logs', action='store_true', help='save Spark logs')
    parser.add_argument('--output_filename', metavar='outputFilename', default='ls.out', help='filename of the output file (default: ls.out)')
    parser.add_argument('--load_N', action='store_true', help='load N')
    parser.add_argument('--save_N', action='store_true', help='save N')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    
    if len(argv)>0 and argv[0]=='print_help':
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)
    (m,n) = args.dims

    # validating
    if m < n:
        raise ValueError('Number of rows({0}) should be greater than number of columns({1})'.format(m,n))

    if args.sketch_type == 'sampling' and args.s is None:
        raise ValueError('Please enter a sampling size!')

    if args.solver_type == 'high_precision' and args.q is None:
        raise ValueError('Please enter number of iterations!')

    if args.solver_type == 'low_precision' and args.sketch_type is None:
        raise ValueError('Please specify a sketch method for the low-precision solver!')

    if args.sketch_type and args.r is None:
        raise ValueError('Please enter a projection size!')

    # loading configuration file
    config = ConfigParser.RawConfigParser()
    config.read(args.setting_filename)

    data_dir = config.get('local_directories','data_dir')
    spark_logs_dir = 'file://'+os.path.dirname(os.path.abspath(__file__))+'/'+config.get('local_directories','spark_logs_dir')
    
    logging.config.fileConfig(args.logging_filename, disable_existing_loggers=False) # setting up the logger
    logger = logging.getLogger('') #using root

    print_params(args, logger) # printing parameters
 
    # instantializing a Spark instance
    if args.save_logs:
        conf = SparkConf().set('spark.eventLog.enabled','true').set('spark.eventLog.dir',spark_logs_dir).set('spark.driver.maxResultSize', '20g')
    else:
        conf = SparkConf().set('spark.driver.maxResultSize', '20g')
    sc = SparkContext(appName="ls_exp",conf=conf)

    if args.file_source=='hdfs':
        hdfs_dir = config.get('hdfs','hdfs_dir')
        Ab_rdd = sc.textFile(hdfs_dir+args.dataset+'.txt',args.npartitions) #loading dataset from HDFS
    elif args.file_source=='s3':
        s3_dir = config.get('s3','s3_dir')
        key_id = config.get('s3','key_id')
        secret_key = config.get('s3','secret_key')
        Ab_rdd = sc.textFile('s3n://'+key_id+':'+secret_key+'@'+s3_dir+args.dataset+'.txt',args.npartitions)
    else:
        A = np.loadtxt(data_dir+args.dataset+'.txt') #loading dataset from local disc
        Ab_rdd = sc.parallelize(A.tolist(),args.npartitions)

    matrix_Ab = RowMatrix(Ab_rdd,args.dataset,m,n,args.cache,stack_type=args.stack_type,repnum=args.nrepetitions) # creating a RowMatrix instance

    ls = RandLeastSquares(matrix_Ab,solver_type=args.solver_type,sketch_type=args.sketch_type,projection_type=args.projection_type,c=args.r,s=args.s,num_iters=args.q,k=args.k)

    ls.fit(args.load_N, args.save_N,args.debug) # solving the problem

    result = {'time':ls.time, 'x':ls.x}
    pickle_write('../result/'+args.output_filename,result) # saving results

    logger.info('Total time elapsed:{0}'.format( ls.time ))

    if args.test:  #only need to load these in the test mode
        if os.path.isfile(data_dir+args.dataset+'_x_opt.txt'):
            logger.info('Found precomputed optimal solutions!')
            x_opt = np.loadtxt(data_dir+args.dataset+'_x_opt.txt')
            f_opt = np.loadtxt(data_dir+args.dataset+'_f_opt.txt')
        else:
            logger.info('Computing optimal solutions!')
            Ab = np.array(matrix_Ab.rdd_original.values().collect()) # might not be accurate, needed to check
            A = Ab[:,:-1]
            b = Ab[:,-1]
            x_opt = np.linalg.lstsq(A,b)[0]
            f_opt = np.linalg.norm(np.dot(A,x_opt)-b)
    
        rx, rf = ls.comp_relerr(x_opt,f_opt)

        logger.info('Median of the relative error on solution vector is:{0}'.format(rx))
        logger.info('Median of the relative error on objective value is:{0}'.format(rf))
    
if __name__ == "__main__":
    main(sys.argv[1:])
