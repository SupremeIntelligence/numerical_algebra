from linear_algebra import input_matrix_from_file, transpose_matrix, multiply_matrix, determinant, trace, krilov_method, show_matrix, iterative_method_of_rotations

A = input_matrix_from_file("input3.txt")
A_T = transpose_matrix(A)
ATA = multiply_matrix(A_T, A)
n = len(A)
show_matrix(ATA)
c0 = [1, 0, 0, 0, 0,]
q = krilov_method(ATA, c0)

print(f"Вектор q: {q}")
ATA_det = determinant(ATA)
print(f"Определитель ATA: {ATA_det}")
ATA_trace = trace(ATA)
print(f"След ATA: {ATA_trace}")
print(f"""Характеристический многочлен матрицы ATA:
P(λ) = λ^{5} + {q[0]:.4}*λ^{4} + {q[1]:.4}*λ^{3} + {q[2]:.4}*λ^{2} + {q[3]:.4}*λ + {q[4]:.4}""")

eigenvalues, eigenvectors, iter_count = iterative_method_of_rotations(ATA, eps=1e-5)

phi = []
for value in eigenvalues:
    p_val = value**5 - q[0]*value**4 - q[1]*value**3 - q[2]*value**2 - q[3]*value - q[4]
    phi.append(p_val)

print("Проверка собственных значений через характеристический многочлен:")
for i in range (n):
    print(f"P(λ{i+1}) = {phi[i]}")



    








