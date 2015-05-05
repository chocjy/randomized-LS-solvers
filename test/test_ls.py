import sys
sys.path.append('../src/')
import numpy as np
import unittest
from projections import Projections
from rowmatrix import RowMatrix
from comp_sketch import comp_sketch

class MatrixMultiplicationTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_Ab = RowMatrix(matrix_rdd,'test_data',1000,10)

    def test_mat_rtimes(self):
        vec = np.random.rand(10)
        p = self.matrix_Ab.rtimes_vec(vec)
        p_true = np.dot( A, vec )

        self.assertTrue( np.linalg.norm(p-p_true) < 1e-5 )

    def test_mat_ltimes(self):
        vec = np.random.rand(1000)
        p = self.matrix_Ab.ltimes_vec(vec)
        p_true = np.dot( vec, A )

        self.assertTrue( np.linalg.norm(p-p_true) < 1e-5 )

    def test_get_b(self):
        b = self.matrix_Ab.get_b()
        
        self.assertTrue( np.linalg.norm(b - Ab[:,-1]) < 1e-5 )

class ProjectionTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_Ab = RowMatrix(matrix_rdd,'test_data',1000,10)

    def test_projection_Gaussian_x(self):
        proj = Projections(projection_type='gaussian',sc=sc,c=1e2,k=3)
        sol = proj.execute(self.matrix_Ab, 'x')
        self.assertEqual(len(sol), 3)
        self.assertEqual(len(sol[0]), self.matrix_Ab.n)

    def test_projection_CW_x(self):
        proj = Projections(projection_type='cw',sc=sc,c=1e2)
        sol = proj.execute(self.matrix_Ab, 'x')
        self.assertEqual(len(sol[0]), self.matrix_Ab.n)

    def test_projection_Rademacher_x(self):
        proj = Projections(projection_type='rademacher',sc=sc,c=1e2)
        sol = proj.execute(self.matrix_Ab, 'x')
        self.assertEqual(len(sol[0]), self.matrix_Ab.n)

    def test_projection_SRDHT_x(self):
        proj = Projections(projection_type='srdht',sc=sc,c=1e2)
        sol = proj.execute(self.matrix_Ab, 'x')
        self.assertEqual(len(sol[0]), self.matrix_Ab.n)

    def test_projection_Gaussian_N(self):
        proj = Projections(projection_type='gaussian',sc=sc,c=1e2,k=3)
        N = proj.execute(self.matrix_Ab, 'N')
        self.assertEqual(len(N), 3)
        self.assertEqual(N[0].shape, (self.matrix_Ab.n,self.matrix_Ab.n))

    def tearDown(self):
        pass

class SketchTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_Ab = RowMatrix(matrix_rdd,'test_data',1000,10)
        self.N_dire = 'N/'

    def test_sketch_projection_x(self):
        x, time = comp_sketch(self.matrix_Ab, 'x', load_N=False, save_N=False, N_dire='N_file/', sketch_type='projection', projection_type='gaussian', k=3, c=1e2)
        self.assertEqual(len(x), 3)
        self.assertEqual(len(x[0]), self.matrix_Ab.n)

    def test_sketch_sampling_x(self):
        x, time = comp_sketch(self.matrix_Ab, 'x', load_N=False, save_N=False, N_dire='N_file/', sketch_type='sampling', projection_type='gaussian', k=3, c=1e2, s=5e2)
        self.assertEqual(len(x), 3)
        self.assertEqual(len(x[0]), self.matrix_Ab.n)

    def test_sketch_sampling_x2(self):
        x, time = comp_sketch(self.matrix_Ab, 'x', load_N=False, save_N=True, N_dire='N_file/', sketch_type='sampling', projection_type='gaussian', k=3, c=1e2, s=5e2)
        self.assertEqual(len(x), 3)
        self.assertEqual(len(x[0]), self.matrix_Ab.n)

    def test_sketch_sampling_x3(self):
        x, time = comp_sketch(self.matrix_Ab, 'x', load_N=True, save_N=True, N_dire='N_file/', sketch_type='sampling', projection_type='gaussian', k=3, c=1e2, s=5e2)
        self.assertEqual(len(x), 3)
        self.assertEqual(len(x[0]), self.matrix_Ab.n)

    def test_sketch_projection_N(self):
        N, time = comp_sketch(self.matrix_Ab, 'N', load_N=False, save_N=False, N_dire='N_file/', sketch_type='projection', projection_type='gaussian', k=3, c=1e2)
        self.assertEqual(len(N), 3)
        self.assertEqual(N[0].shape, (self.matrix_Ab.n,self.matrix_Ab.n))

    def test_sketch_sampling_N(self):
        N, time = comp_sketch(self.matrix_Ab, 'N', load_N=False, save_N=False, N_dire='N_file/', sketch_type='projection', projection_type='gaussian', k=3, c=1e2, s=5e2)
        self.assertEqual(len(N), 3)
        self.assertEqual(N[0].shape, (self.matrix_Ab.n,self.matrix_Ab.n))

    def test_sketch_sampling_N2(self):
        N, time = comp_sketch(self.matrix_Ab, 'N', load_N=True, save_N=False, N_dire='N_file/', sketch_type='projection', projection_type='gaussian', k=3, c=1e2, s=5e2)
        self.assertEqual(len(N), 3)
        self.assertEqual(N[0].shape, (self.matrix_Ab.n,self.matrix_Ab.n))

loader = unittest.TestLoader()
suite_list = []
suite_list.append( loader.loadTestsFromTestCase(MatrixMultiplicationTestCase) )
suite_list.append( loader.loadTestsFromTestCase(ProjectionTestCase) )
suite_list.append( loader.loadTestsFromTestCase(SketchTestCase) )
suite = unittest.TestSuite(suite_list)

if __name__ == '__main__':
    from pyspark import SparkContext

    sc = SparkContext(appName="ls_test_exp")
    Ab = np.loadtxt('../data/nonunif_bad_1000_10.txt')
    A = Ab[:,:-1]
    matrix_rdd = sc.parallelize(Ab.tolist(),140)
    #stream=sys.stderr, descriptions=True, verbosity=1

    runner = unittest.TextTestRunner(stream=sys.stderr, descriptions=True, verbosity=1)
    runner.run(suite)

