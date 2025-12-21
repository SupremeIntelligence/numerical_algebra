from linear_algebra import input_matrix_from_file, show_matrix, gaussian_method, multiply_matrix, cube_norm, left_sweep_method

filename = "input2.txt"
A = input_matrix_from_file(filename)
show_matrix(A)

#Решение СЛАУ
X = []
X = left_sweep_method(A)
print (f"Решение СЛАУ методом прогонки: {X}")

x_gauss, det = gaussian_method(A)

print (f"Решение методом Гаусса: {x_gauss}")

A = input_matrix_from_file("input2.txt")

# Вычисление r
A = input_matrix_from_file(filename)
n = len(A)
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

A = input_matrix_from_file(filename)

X_t = [[x_gauss[i]] for i in range (n)]

for row in A:
    b.append(row[-1])
    row.pop()
    
r = multiply_matrix(A, X_t)
for i in range (n):
    r[i][0] -=b[i]

print("Вычисление r для метода Гаусса:" )
show_matrix(r, 4, True)

#Вычисление нормы r
print(f"Норма r для метода Гаусса: {cube_norm(r)}")






    








