from linear_algebra import input_matrix_from_file, transpose_matrix, transpose, multiply_matrix, show_matrix, sign, get_proximity_measure, find_max_off_diagonal_element, cube_norm, rotation_matrix, iterative_method_of_rotations
import numpy as np
import math 

A = input_matrix_from_file("input3.txt")
A_T = transpose_matrix(A)
ATA = multiply_matrix(A_T, A)
eps=1e-5
n = len(A)

ATA_orig = [row[:] for row in ATA]
show_matrix(ATA)

eigenvalues, eigenvectors, iter_count = iterative_method_of_rotations(ATA, eps=eps)

print("Собственные значения:")
for i in range (n):
    print(f"λ{i+1} = {eigenvalues[i]}")

print("Собственные векторы:")
show_matrix (eigenvectors)

q = 1 - 2/(n*(n-1))
t_A = get_proximity_measure(ATA_orig)
k = (math.log(eps) - math.log(t_A)) / math.log(q)
print(f"Априорная оценка количества итераций: {math.ceil(k)}")
print(f"Количество итераций: {iter_count}")

for k in range (n):
    r = multiply_matrix(ATA_orig, transpose(eigenvectors[k]))
    for i in range (n):
        r[i][0] -= eigenvalues[k]*eigenvectors[k][i]

    print(f"Невязка для собственного вектора {k+1}:")
    show_matrix(r, 10, True)
    r_norm = cube_norm(r)
    print(f"Норма невязки для собственного вектора {k+1}: {r_norm}")
    
"""ATA_np = np.array(ATA_orig)
np_eigenvalues, np_eigenvectors = np.linalg.eigh(ATA_np)
print("\nСобственные значения через numpy:")
print(np_eigenvalues)
print("Собственные векторы через numpy (столбцы матрицы):")
print(np_eigenvectors)"""





