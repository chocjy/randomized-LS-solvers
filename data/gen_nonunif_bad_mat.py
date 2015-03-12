"""
Generating matrices with nonuniform leverage scors and bad condition number.
usage:
  python gen_nonunif_bad_mat.py m n [test]

parameters:
m, n: the size of the matrix to be generated
test: whether to compute the optimal solution or not
"""

import numpy as np
import sys

def gen_nonunif_bad_mat(m,n):
    a1 = np.hstack((50*np.random.randn(m-n/2,n/2),1e-8*np.random.randn(m-n/2,n/2)))
    a2 = np.hstack((np.zeros((n/2,n/2)),np.eye(n/2)))
    A = np.vstack((a1,a2))
    x = np.random.randn(n)
    b = np.dot(A,x)
    err = np.random.randn(m)
    b = b + 0.25*np.linalg.norm(b)/np.linalg.norm(err)*err

    return A,b

if __name__ == "__main__":
    m = int(sys.argv[1])
    n = int(sys.argv[2])

    filename = 'nonunif_bad_'+str(m)+'_'+str(n)

    A,b = gen_nonunif_bad_mat(m,n)
    Ab = np.hstack((A,b.reshape((m,1))))
    np.savetxt(filename+'.txt',Ab,fmt='%.12e')

    if len(sys.argv) > 3:
        x_opt = np.linalg.lstsq(A,b)[0]
        f_opt = np.linalg.norm(np.dot(A,x_opt)-b)

        np.savetxt(filename+'_x_opt.txt',x_opt)
        np.savetxt(filename+'_f_opt.txt',np.array([f_opt]))

