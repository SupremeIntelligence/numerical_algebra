import numpy as np
import copy
from linear_algebra import input_matrix_from_file, show_matrix, gaussian_method, multiply_matrix, cube_norm

filename = "input.txt"
A = input_matrix_from_file(filename)
show_matrix(A)

#Решение СЛАУ
X = []
X, determinator = gaussian_method(A)
print (f"Решение СЛАУ: {X}")

# Проверка решения через NumPy
A = input_matrix_from_file(filename)
A_np = np.array([row[:-1] for row in A], dtype=float)
B_np = np.array([row[-1] for row in A], dtype=float)

x_np = np.linalg.solve(A_np, B_np)
print("Проверка через NumPy:", x_np)

print (f"Определитель матрицы: {determinator}")
print (f"Проверка определителя через NumPy: {np.linalg.det(A_np)}")

# Поиск обратной матрицы
for row in A:
    row[-1] = 0.0

A_orig = copy.deepcopy(A)
n = len(A)

inversed_A = []
for i in range (n):
    A = copy.deepcopy(A_orig)
    A[i][-1] = 1.0
    #show_matrix(A)
    X_col = gaussian_method(A)[0]
    inversed_A.append(X_col)

# Транспонирование матрицы
inversed_A = list(map(list, zip(*inversed_A)))
print ("Обратная матрица:")
show_matrix(inversed_A, 10)

#Вычисление числа обусловенности
cond_numb=cube_norm(A_orig) * cube_norm(inversed_A)
print(f"Число обусловленности матрицы A: {cond_numb}")

#Вычисление R
A = copy.deepcopy(A_orig)
for row in A:
    row.pop()

R = multiply_matrix(inversed_A, A)
for i in range (n):
    R[i][i] -= 1.0
print("Вычисление R:" )
show_matrix(R, 4, True)

#Проверка обратной матрицы и R через NumPy
A = copy.deepcopy(A_orig)
A_np = np.array([row[:-1] for row in A], dtype=float)
inversed_A_np = np.linalg.inv(A_np)
print("Проверка обратной матрицы через NumPy:")
show_matrix(inversed_A_np)
R_np = np.dot(inversed_A_np, A_np) - np.eye(A_np.shape[0])
print("Проверка R через NumPy:")
show_matrix(R_np, 4, True)

# Вычисление r
A = input_matrix_from_file(filename)
b = []
X_t = [[X[i]] for i in range (n)]

for row in A:
    b.append(row[-1])
    row.pop()
    
r = multiply_matrix(A, X_t)
for i in range (n):
    r[i][0] -=b[i]

print("Вычисление r:" )
show_matrix(r, 4, True)

#Вычисление нормы r
print(f"Норма r: {cube_norm(r)}")

#Вычисление нормы R
print(f"Норма R: {cube_norm(R)}")






    








