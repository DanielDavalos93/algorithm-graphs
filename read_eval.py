import numpy as np 
import isomorphisms as ism 
import wdg

arrs = []

n = int(input("n:\n"))

"""with open(f"all_matrices_{n}.csv", "r") as file:
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
    matrices.append(np.array(matrix))  # Convertir a NumPy y agregar a la lista
"""

def load_matrices(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    matrices = []
    for line in lines:
        cleaned_line = line.strip().replace("[", "").replace("]", "")  # Elimina corchetes
        rows = [r.strip() for r in cleaned_line.split(",") if r.strip()]  # Divide por filas
        
        # Convertir cada fila en una lista de números
        matrix = np.array([list(map(float, row.replace(",", "").split())) for row in rows])
        matrices.append(matrix)
    
    return np.array(matrices,dtype=object)

# Convertir la lista de matrices a un array de NumPy (opcional)
result = load_matrices(f"all_matrices_{n}.csv")

#class1 = ism.partition_only_pos(result, ism.are_related)

#classes = list(map(lambda x: list(filter(lambda y: wdg.weight(y) == max(list(map(wdg.weight, x))),x)) , class1))

def comparar(representante, i_class):
    i = 0
    for m in classes[i_class]:
        print(i,": es permutable? : ", ism.are_perm(representante, m))
        i += 1

"""
Las clases de equivalencia que contienen al menor delta en K6:

"""

def remove_duplicates(array):
  seen = {}
  new_array = []
  for element in array:
    if element not in seen:
      seen[element] = 1
      new_array.append(element)
  return new_array

#all_deltas = remove_duplicates(list(map(lambda x: wdg.delta(x), matrices)))

def matrix_repr(n):
    if n == 5:
        #m = [[ 0,  1,  1,  1,  1],
        #    [ 1,  0,  1,  1, -1],
        #    [ 1,  1,  0, -1,  1],
        #    [ 1,  1, -1,  0,  1],
        #    [ 1, -1,  1,  1,  0]]  
        m = wdg.circ(5)
    elif n == 6:
        m = np.array([[ 0,  1,  1,  1,  1,  1],
                    [ 1,  0, -1,  1,  1, -1],
                    [ 1, -1,  0, -1,  1,  1], 
                    [ 1,  1, -1,  0, -1,  1],
                    [ 1,  1,  1, -1,  0, -1],
                    [ 1, -1,  1,  1, -1,  0]])
    elif n == 7:
        m = [[ 0,  1,  1,  1,  1,  1,  1],
            [ 1,  0,  1,  1,  1,  1, -1],
            [ 1,  1,  0,  1,  1, -1,  1],
            [ 1,  1,  1,  0, -1,  1,  1],
            [ 1,  1,  1, -1,  0, -1, -1],
            [ 1,  1, -1,  1, -1,  0, -1],
            [ 1, -1,  1,  1, -1, -1,  0]]


    return m
"""
n=7 :
    m = np.array([[ 0, -1, -1,  1,  1, -1, -1], 
                      [-1,  0,  1,  1, -1, -1, -1], 
                      [-1,  1,  0,  1,  1,  1, -1], 
                      [ 1,  1,  1,  0,  1, -1,  1],  
                      [ 1, -1,  1,  1,  0,  1,  1], 
                      [-1, -1,  1, -1,  1,  0,  1], 
                      [-1, -1, -1,  1,  1,  1,  0]])
"""
repr = matrix_repr(n)

#print("delta:",best_delta)

if n == 5:
    best_delta = 8
elif n == 6:
    best_delta = 10
elif n == 7:
    best_delta = 16
else:
    best_delta = min(all_deltas)


