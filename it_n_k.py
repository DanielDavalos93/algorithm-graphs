import isomorphisms as ism 
import numpy as np
import csv

n = int(input("Ingrese tamaño n:\n"))

k = int()

conj = ism.gen_matrices_k_neg(n,k)

filename = 'matrices_%s.csv' % n
f = open(filename, "a")

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',lineterminator='\n')
    for m in conj:
        writer.writerow(m)
    #writer.writerow('\n')
    #for m in conj:
    #f.write(np.array(m))
#for m in conj:
#    np.savetxt(filename,m,delimiter=',')

f.close()
