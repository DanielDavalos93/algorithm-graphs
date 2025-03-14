import wdg
import numpy as np
from itertools import product,permutations
from functools import lru_cache

##


@lru_cache(maxsize=None)
def cached_delta(matrix):
    return wdg.delta(matrix)

def permutator_row(permutator):
  N = np.empty((len(permutator),len(permutator)))
  M = np.diag(np.ones(len(permutator)))
  for i in range(len(M)):
    N[:,i] = M[:,permutator[i]]
  return N

def permutator_column(permutator):
  N = np.empty((len(permutator),len(permutator)))
  M = np.diag(np.ones(len(permutator)))
  for i in range(len(M)):
    N[i] = M[permutator[i]]
  return N 


def permutation(M,permutator):
  return np.matmul(permutator_row(permutator),np.matmul(M,permutator_column(permutator)))

def are_perm(M,N):
    n = len(M)
    res = False
    for perm in permutations(range(n)):
        if np.array_equal(N,permutation(M,perm)) or np.array_equal(M,N):
           return True 
    return False


def swap(M,ls):
  m_swap = np.diag(np.ones(len(M)))
  for i in range(len(M)):
    m_swap[i][i] = ls[i]
  return np.matmul(np.matmul(m_swap,M),m_swap)

def accesible(G1,G2):
    """
    Dos matrices G1 y G2 son isomorfas si existe una matriz
    diagonal D (que representa el valor de sus vértices) tal que 
    G1 = D x G2 x D.
    """
    n = len(G1)
    if len(G2) != n:
        return False 
    else:
        res = False
        V = list(product([1,-1],repeat=n))
        for v in V:
            d = np.diag(v)
            p = np.matmul(np.matmul(d,G1),d)
            res = res | np.array_equal(G2,p)
            if res:
                break
            else:
                continue 
        return res

def eq_class(a,rel,A):
    return list(filter(lambda x: rel(x,a),A))

def partition(a, equiv):
    partitions = [] 
    for e in a:         
        found = False 
        for p in partitions:
            if equiv(e, p[0]): 
                p.append(e)
                found = True
                break
        if not found: 
            partitions.append([e])
    return partitions

def partition_only_pos(a, equiv):
    partitions = []
    seen = set()
    for e in a:
        if wdg.weight(e) > 0:
            # Convertimos la matriz en una representación hashable (tuple de tuples)
            e_tuple = tuple(map(tuple, e))
            if e_tuple in seen:
                continue
            
            found = False
            for p in partitions:
                if equiv(e, p[0]):
                    p.append(e)
                    found = True
                    break

            if not found:
                partitions.append([e])
                seen.add(e_tuple) 
    return partitions

#Toma un representante de la clase de equiv.
def repr_eq_classes(ls, f):
    processed = [False] * len(ls)
    representatives = []
    for i in range(len(ls)):
        if not processed[i]:
            current = ls[i]
            representatives.append(current)
            for j in range(len(ls)):
                if not processed[j] and f(current, ls[j]):
                    processed[j] = True
    res = list(map(lambda x: (x,wdg.delta(x)), representatives))
    return res

#Relation
def are_related(M,N):
    n = len(M)
    signs = np.array(list(product([1, -1], repeat=n)))
    # Generamos todas las matrices transformadas por signos
    diag_matrices = np.einsum('ij,jk,ik->ijk', signs, M, signs)  # Broadcasting

    return np.any(np.all(diag_matrices == N, axis=(1, 2))) or np.array_equal(M, N)


def generate_matrices(n):
    # Generate all combinations for the upper triangular part (excluding the diagonal)
    matrices = []
    combinations = wdg.combinations_maj_pos(n)

    for comb in combinations:
        # Initialize an n x n matrix with zeros
        matrix = np.zeros((n,n))
        index = 0
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j] = comb[index]
                matrix[j][i] = comb[index]
                index += 1
        matrices.append((matrix,wdg.delta(matrix)))
    #return list(filter(lambda x: wdg.weight(x) > 0, matrices))
    return matrices

def generate_all_matrices(n):
    # Generate all combinations for the upper triangular part (excluding the diagonal)
    matrices = []
    combinations = wdg.combinations_all(n)

    for comb in combinations:
        # Initialize an n x n matrix with zeros
        matrix = np.zeros((n,n))
        index = 0
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j] = comb[index]
                matrix[j][i] = comb[index]
                index += 1
        matrices.append(matrix)
    #return list(filter(lambda x: wdg.weight(x) > 0, matrices))
    return matrices

def gen_matrices_k_neg(n,k):
    matrices = []
    mid = (len(wdg.m1_len(n,k))+1) // 2
    for comb in wdg.m1_len(n,k)[:mid]:
        # Initialize an n x n matrix with zeros
        matrix = np.zeros((n,n))

        for i in range(n):
            for j in range(i + 1, n):
                val_ij = comb[i]*comb[j]
                matrix[i][j] = val_ij
                matrix[j][i] = val_ij

        matrices.append(matrix)
    #return list(filter(lambda x: wdg.weight(x) > 0, matrices))
    return matrices

def matrix_k_pos(n,k):
    matrices = []
    tr_n = (n*(n-1)) // 2
    for comb in wdg.m1_len(tr_n,k):
        matrix = np.zeros((n,n))
        index = 0
        for i in range(n):
            #index = 0
            for j in range(i+1,n):
                matrix[i][j],matrix[j][i] = comb[index],comb[index]
                index += 1 
        if wdg.weight(matrix) > 0:
            matrices.append(matrix)
    return matrices

def equiv_class(M):
  eq = [M]
  for i in range(len(M)):
    for j in range(i+1,len(M)):
      eq.append(swap(M,i,j))
  return eq


def eval_diff(matrices):
    for m in matrices:
        #maxv = max(m)
        #minv = min(m)
        #print(f"{k} max:{maxv} \t min:{minv}")
        print(f"delta:{wdg.delta(m)}")

def best_matrix(lm):
  n = len(lm[0])
  matrix = wdg.K(n)
  for mtx in lm(n)[:len(lm(n))//2]:
    if delta(mtx) < delta(matrix):
      matrix = mtx
  print(f"best matrix order {n}: \n{matrix} \t\t delta : {wdg.delta(matrix)}")

# equivalentes y permutables
def permutable_and_equivalent(M,N):
    n = len(M)
    res = False
    for perm in permutations(list(range(n))):
        res = res | are_related(N,permutation(M,perm)) | np.array_equal(M,N)
        if res:
            break
        else:
            continue
    return res 
