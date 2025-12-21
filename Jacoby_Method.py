import numpy as np
from linear_algebra import show_matrix, input_matrix_from_file, jacobi_method, check_jacobi_convergence, cube_norm

A = input_matrix_from_file("input.txt")
b = []
eps = 1e-5
for row in A:
    b.append(row[-1])
    row.pop()

if check_jacobi_convergence(A):
    print("\n Метод Якоби будет сходиться для данной системы.")
else:
    print("\nУсловия сходимости метода Якоби не выполняются. Сходимость не гарантируется.")

x, iter_count, r = jacobi_method(A, b, b, eps)  

print(f"\nРешение СЛАУ: x = {x}")
print(f"\nКоличество итераций: {iter_count}")
print("\nНевязка r:")
show_matrix(r, 10, True)
print(f"\nНорма r: {cube_norm(r)}")
