import isomorphisms as ism
import wdg
import numpy as np
import csv
from multiprocessing import Pool

def generate_and_write_matrix(args):
    n, i = args
    matrix = ism.generate_all_matrices(n)[i]
    return matrix.flatten()

n = int(input("Ingrese tamaño n:\n"))

#conj = np.array(ism.generate_all_matrices(n), datatype=object)

filename = f'all_matrices_{n}.csv'

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',lineterminator='\n')
    with Pool() as pool:
        matrices = pool.map(generate_and_write_matrix, [(n,i) for i in range(len(ism.generate_all_matrices(n)))])
    for matrix in matrices:
        writer.writerow(matrix)


