import numpy as np
from itertools import product,permutations
from functools import lru_cache

# An undi weighted graph is a symmetric matrix

def get_value(l,value):
    return filter(lambda x: x==value, l)

def tuple_less(tuples):
    new_tuples = []
    for tp in tuples:
        for i in range(len(tp)):
            for j in range(i,len(tp)):
                if tp[i] < tp[j]:
                    new_tuples.append(tp)
                else:
                    continue
    return new_tuples

def m1_len(n,k):
    filt = filter(lambda l: len(list(get_value(l,1)))==k, product([-1,1],repeat=n))
    return np.array(list(filt))

def combinations_maj_pos(n):
    tr = (n*(n-1))//2
    finish = 2**(tr-1)
    combinations = np.array(list(product([1, -1], repeat=tr))[:finish])
    return combinations 

def combinations_all(n):
    tr = (n*(n-1))//2
    combinations = np.array(list(product([1, -1], repeat=tr)))
    return combinations 

#Functions
def g(matrix,x):
    v = np.array(x)
    return float(np.matmul(np.matmul(v,matrix),v.T)/2)

def weight(matrix):
    #lst = list(filter(lambda x: x[0]<x[1], product(range(len(matrix)),repeat=2)))
    #return sum([matrix[i][j] for (i,j) in lst])
    x = np.array([[1]*(len(matrix))])
    return g(matrix,x)

def max_g(matrix,vecs):
    return max([g(matrix,x) for x in vecs])

def min_g(matrix,vecs):
    return min([g(matrix,x) for x in vecs])

def delta(matrix):
    vecs = list(map(list,product([-1,1],repeat=len(matrix))))
#    vecs =combinations_maj_pos(len(matrix)) 
# return float(max_g(matrix,vecs) - min_g(matrix,vecs))
    return max([g(matrix,x) for x in vecs]) - min([g(matrix,x) for x in vecs])



"""
Tipos de grafos:

* Grafos completos de orden n
"""
def K(n):
  M = np.ones((n,n))
  for i in range(n):
    M[i][i] = 0
  return M

"""
* Grafos circulares de orden n
"""
def circ(n):
  M = K(n)
  M[0][n-1] = -1
  M[n-1][0] = -1
  for i in range(n-1):
    M[i][i+1] = -1
    M[i+1][i] = -1
  return M

"""
* Grafos aleatorios de orden n
"""
def random_symm_matrix(n):
  np.set_printoptions(precision=3)
  b = np.random.randn(n,n)/5
  b_symm = (b + b.T)/2
  for i in range(n):
     b_symm[i][i] = 0
  return b_symm
