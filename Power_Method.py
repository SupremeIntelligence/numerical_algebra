from linear_algebra import input_matrix_from_file, transpose_matrix, multiply_matrix, show_matrix, cube_norm, transpose, power_method

A = input_matrix_from_file("input3.txt")
A_T = transpose_matrix(A)
ATA = multiply_matrix(A_T, A)
n = len(A)
show_matrix(ATA)

eps=1e-5

min_eigenvalue, y_new, iter_count  = power_method(ATA, eps=eps)
print(f"Собственное значение: λ = {min_eigenvalue}")
print(f"Собственный вектор: ")
show_matrix([y_new])
print(f"Количество итераций: {iter_count}")

r = multiply_matrix(ATA, transpose(y_new))
for i in range (n):
    r[i][0] -= min_eigenvalue*y_new[i]
print("Невязка для собственного вектора:")
show_matrix(r, 10, True)
r_norm = max(abs(v) for v in y_new)
print(f"Кубическая норма невязки для собственного вектора: {r_norm}")
r_norm = cube_norm(r)
print(f"Норма невязки для собственного вектора: {r_norm}")


