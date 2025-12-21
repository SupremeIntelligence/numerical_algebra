from linear_algebra import input_matrix_from_file, transpose_matrix, multiply_matrix, show_matrix, determinant, trace, iterative_method_of_rotations, transpose, cube_norm, danilevski_method

A = input_matrix_from_file("input3.txt")
A_T = transpose_matrix(A)
ATA = multiply_matrix(A_T, A)
ATA_orig = [row[:] for row in ATA]
n = len(A)
show_matrix(ATA)

F, S = danilevski_method(ATA)

print("Матрица Фробениуса для ATA:")
show_matrix(F)

ATA_det = determinant(ATA)
print(f"Определитель ATA: {ATA_det}")
ATA_trace = trace(ATA)
print(f"След ATA: {ATA_trace}")

eigenvalues, eigenvectors, iter_count = iterative_method_of_rotations(ATA, eps=1e-5)

print("Собственные значения, полученные методом вращений:")
for i in range (n):
    print(f"λ{i+1} = {eigenvalues[i]}")

print("Матрица собственных векторов, полученная методом вращений:")
show_matrix (eigenvectors)

eigenvectors = []
for value in eigenvalues:
    F_vector = [value**(n-i-1) for i in range(n)]
    #print (f"Собственный вектор матрицы Фробениуса для собственного значения {value}: {F_vector}")
    F_vector_col = [[F_vector[i]] for i in range (n)]
    vector = multiply_matrix(S, F_vector_col)
    vector = [vector[i][0] for i in range (n)]
    eigenvectors.append(vector)

print("Матрица собственных векторов, полученная через матрицу Фробениуса:")
show_matrix (eigenvectors)

for k in range (n):
    r = multiply_matrix(ATA_orig, transpose(eigenvectors[k]))
    for i in range (n):
        r[i][0] -= eigenvalues[k]*eigenvectors[k][i]
    print(f"Невязка для собственного вектора {k+1}:")
    show_matrix(r, 10, True)
    r_norm = cube_norm(r)
    print(f"Норма невязки для собственного вектора {k+1}: {r_norm}")