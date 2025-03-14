import isomorphisms as ism 
import numpy as np
import csv

arrs = []

n = int(input("n:\n"))

with open(f"all_matrices_{n}.csv", "r") as file:
    lines = file.readlines()

# Procesar cada línea como una matriz
matrices = []
for line in lines:
    # Limpiar la línea y eliminar los caracteres innecesarios
    cleaned_line = line.strip()  # Elimina espacios y saltos de línea
    cleaned_line = cleaned_line.replace("[", "").replace("]", "")  # Elimina corchetes
    rows = cleaned_line.split(",")  # Divide en las filas de la matriz
    # Convertir cada fila en una lista de números
    matrix = [list(map(float, row.split())) for row in rows]
    matrices.append(np.array(matrix)) 

#conj = ism.equivalence_classes(matrices,ism.accesible)
conj = np.array(ism.partition_only_pos(matrices, ism.are_related), dtype=object)

filename = 'equiv_delta_%s.csv' % n
f = open(filename, "a")

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',lineterminator='\n')
    for m in conj:
        writer.writerow(m)
        writer.writerow('\n')
    #writer.writerow('\n')
    #for m in conj:
    #f.write(np.array(m))
#for m in conj:
#    np.savetxt(filename,m,delimiter=',')


f.close()
