"""Модуль для работы с линейной алгеброй: ввод матриц, методы решения СЛАУ, вычисление собственных значений и векторов, и другие операции с матрицами и векторами."""

def input_matrix_from_file(filename: str = "input.txt") -> list:
    """Ввод расширенной матрицы A|b из файла."""
    with open (filename, "r", encoding="utf-8") as file:
        matrix = []
        for line in file:
            row = list(map(float, line.strip().split()))
            matrix.append(row)
    return matrix

def show_matrix (matrix: list[list], precision: int = 4, e: bool = False) -> None:
    """Вывод матрицы."""
    if e:
        for row in matrix:
            print(" ".join(f"{value:8.{precision}e}" for value in row))
    else:
        for row in matrix:
            print(" ".join(f"{value:8.{precision}f}" for value in row))
    

def find_max_element(matrix: list[list], start_row: int = 0, start_col: int = 0) -> float:
    """Поиск главного элемента по матрице."""
    n = len(matrix)
    m = len(matrix[0]) - 1
    max_val = abs(matrix[start_row][start_col])
    row_index = start_row
    col_index = start_col

    for i in range(start_row, n):
        for j in range(start_col, m):
            if abs(matrix[i][j]) > max_val:
                max_val = abs(matrix[i][j])
                row_index = i
                col_index = j
    return max_val, row_index, col_index

def swap_rows(matrix: list[list], row1: int, row2: int) -> None:
    """Перестановка строк матрицы."""
    matrix[row1], matrix[row2] = matrix[row2], matrix[row1]

def swap_columns(matrix: list[list], col1: int, col2: int)->None:
    """Перестановка столбцов матрицы."""
    for row in matrix:
        row[col1], row[col2] = row[col2], row[col1]

def gaussian_method(A: list[list]) -> tuple[list[float], float]:
    """Решение СЛАУ методом Гаусса с выбором главного элемента по матрице. Возвращает пару (вектор решения, определитель матрицы)."""
    n = len(A)
    order = list (range(n))
    determinator = 1.0

    # Прямой ход метода Гаусса с выбором главного элемента по всей матрице
    for step in range (n):
        pivot, pivot_row, pivot_col = find_max_element(A, step, step)
        if pivot == 0.0:
            raise ValueError("Матрица вырождена, решение не существует")
        #print(f"Максимальный элемент: {pivot} на позиции ({pivot_row}, {pivot_col})") 
        
        determinator *= pivot

        #перестановка строк и столбцов для установки главного элемента на диагональ
        if A[step][step] != pivot:
            if pivot_row != step:
                swap_rows (A, pivot_row, step)
                determinator *= -1
            if pivot_col != step:
                swap_columns (A, pivot_col, step)
                determinator *= -1

            order[pivot_col], order[step] = order[step], order[pivot_col]

            #print(f"После перестановки строк и столбцов для шага {step+1}:")
            #show_matrix(A)

        #print(f"После деления ведущей строки на ведущий элемент для шага {step+1}:")
        A[step] = [value / A[step][step] for value in A[step]]
        #show_matrix(A)

        for i in range (step+1, n):
            factor = A[i][step]                 
            if factor == 0.0:
                continue
            for j in range (step, n+1):
                A[i][j] -= A[step][j] * factor
                #print (f"A[{i}][{j}] -= A[{step}][{j}] * A[{i}][{step}] = {A[i][j]}")
        #print(f"После вычитания для шага {step+1}:")
        #show_matrix(A)

    X = [0] * n
    for i in range (n-1, -1, -1):
        X[i] = A[i][n] - sum(A[i][j]*X[j] for j in range (i+1, n))
        #print (f"X[{i}]=A[{i}][{n}] - {sum(A[i][j]*X[j] for j in range (i+1, n))} = {root}")

    #Восстановление порядка переменных
    X_final = [0] * n
    for i in range (n):
        X_final[order[i]] = X[i]

    return X_final, determinator

def multiply_matrix (A: list[list], B: list[list]) -> list[list]:
    """Умножение матриц A и B."""
    n = len(A) 
    m = len(A[0])
    p = len(B[0]) 

    if len(B) != m:
        raise ValueError("Матрицы нельзя перемножить: число столбцов A != числу строк B")

    C = [[0.0 for _ in range(p)] for _ in range(n)]

    for i in range(n): 
        for j in range(p):  
            for k in range(m):     
                C[i][j] += A[i][k] * B[k][j]

    return C

def multiply_vectors (X: list[float], Y: list[float]) -> float:
    """Вычисление скалярного произведения векторов X и Y."""
    if len(X) != len(Y):
        raise ValueError("Векторы должны быть одинаковой длины для вычисления скалярного произведения.")
    result = sum(X[i] * Y[i] for i in range(len(X)))
    return result

def transpose (X: list[list]) -> list[list]:
    """Транспонирование вектора."""
    return [[X[i]] for i in range (len(X))]

def transpose_matrix (A: list[list]) -> list[list]:
    """Транспонирование матрицы."""
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def cube_norm(A: list[list]) -> float:
    """Вычисление кубической нормы матрицы."""
    norm = 0.0
    for row in A:
        row_sum = sum(abs(value) for value in row)
        if row_sum > norm:
            norm = row_sum
    return norm

def determinant(matrix: list[list]) -> float:
    """
    Вычисление определителя квадратной матрицы рекурсивно.
    matrix: список списков (матрица)
    """
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for col in range(n):
        submatrix = [row[:col] + row[col+1:] for row in matrix[1:]]
        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(submatrix)
    return det

def trace (matrix: list[list]) -> float:
    """Вычисление следа квадратной матрицы."""
    n = len(matrix)
    tr = sum(matrix[i][i] for i in range (n))
    return tr

def sign (x) -> int:
    """Функция сигнум."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def left_sweep_method(A: list[list]) -> list[float]:
    """Решение СЛАУ методом прогонки (методом левой прогонки). Возвращает вектор решения."""
    n = len(A)
    alpha = n*[0.0]
    beta = n*[0.0]
    x = n*[0.0]

    #левая прогонка
    alpha[n-1] = -(A[n-1][n-2]/A[n-1][n-1])
    beta[n-1] = A[n-1][n]/A[n-1][n-1]
    #print (f"alpha[{n-1}] = {alpha[n-1]}")
    #print (f"beta[{n-1}] = {beta[n-1]}")

    for i in range(n-2, -1, -1):
        denominator = A[i][i] + (alpha[i+1]*A[i][i+1])
        if i == 0:
            alpha[i] = 0.0
        else:
            alpha[i] = -(A[i][i-1]/denominator)
        beta[i] = (A[i][n]-(beta[i+1]*A[i][i+1]))/denominator
        
        #print (f"alpha[{i}] = {alpha[i]}")
        #print (f"beta[{i}] = {beta[i]}")

    #обратная прогонка
    x[0] = beta[0]
    #print(f"x[0] = {x[0]}")
    for i in range(0, n-1):
        x[i+1] = alpha[i+1]*x[i] + beta[i+1]
        #print(f"x[{i+1}] = {x[i+1]}")

    #print (alpha, "\n", beta)
    return x

def jacobi_method(A: list[list], b: list[float], init_x: list[float], eps: float = 1e-5) -> tuple[list[float], int, list[list]]:
    """
    Решение системы Ax = b методом Якоби. Возвращает пару (вектор решения, число итераций, невязка).
    """
    n = len(b)
    x = init_x.copy()           
    x_new = [0.0] * n       
    diff = eps + 1
    r_norm = 0

    if init_x is None:
        x = [0.0] * n
    else:
        x = init_x.copy()

    print(f"{'Итерация':<10} " + " ".join([f"x{i+1:<11}" for i in range(n)]) + "||r||")

    iter = 0

    while diff >= eps:
        print (diff)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x[j]
            x_new[i] = (b[i] - s) / A[i][i]

        diff = cube_norm(transpose([x_new[i] - x[i] for i in range(n)]))

        x_t = transpose(x_new)
        b_t = transpose(b)

        Ax = multiply_matrix(A, x_t)
        r = [[Ax[i][0] - b_t[i][0]] for i in range(n)]
        
        r_norm = cube_norm(r)
        #print (r_norm)
        
        print(f"{iter + 1:<10}" + " ".join([f"{x_new[i]:<12.6f}" for i in range(n)]) + f"{r_norm:.2e}")

        iter += 1
        x = x_new.copy()

    return x_new, iter, r

def check_jacobi_convergence(A: list[list]) -> bool:
    """Проверка достаточных условий сходимости метода Якоби."""

    n = len(A)
    #Построчное диагональное преобладание
    row_dom = all(sum(abs(A[i][j]/A[i][i]) for j in range(n) if j != i) < 1 for i in range(n))

    #Постолбцовое диагональное преобладание
    col_dom = all(sum(abs(A[i][j]/A[i][i]) for i in range(n) if i != j) < 1 for j in range(n))

    #Условие суммы квадратов
    third_cond = sum(
        (A[i][j] / A[i][i])**2
        for i in range(n)
        for j in range(n)
        if i != j
    ) < 1

    print("Проверка условий сходимости метода Якоби:")
    print(f"1) {row_dom}")
    print(f"2) {col_dom}")
    print(f"3) {third_cond}")

    return row_dom or col_dom or third_cond
   
def sor_method(A: list[list], b: list[float], omega: float, init_x: list[float], eps: float = 1e-5)->tuple[list[float], int, list[list]]:
    """
    Решение системы Ax = b методом верхней релаксации. Возвращает пару (вектор решения, число итераций, невязка).
    """
    n = len(b)
    if init_x is None:
        init_x = [0.0] * n
    else:
        x = init_x.copy()           
    x_new = [0.0] * n       
    diff = eps+1
    print(f"{'Итерация':<10} " + " ".join([f"x{i+1:<11}" for i in range(n)]) + "||r||")

    iter = 0

    while diff >= eps:
        x = x.copy()

        # Итерации SOR
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j == i:
                    continue
                xj = x_new[j] if j < i else x[j]
                sigma += A[i][j] * xj

            x_new[i] = (1 - omega) * x[i] + (omega / A[i][i]) * (b[i] - sigma)

        diff = cube_norm(transpose([x_new[i] - x[i] for i in range(n)]))

        x = x_new.copy()

        x_t = transpose(x_new)
        b_t = transpose(b)

        Ax = multiply_matrix(A, x_t)
        r = [[Ax[i][0] - b_t[i][0]] for i in range(n)]
        r_norm = cube_norm(r)
        print(f"{iter + 1:<10}" + " ".join([f"{x_new[i]:<12.6f}" for i in range(n)]) + f"{r_norm:.2e}")
        iter+=1

    return x, iter, r

def check_sor_convergence (A: list[list], omega: float) -> bool:
    """Проверка условий сходимости метода верхней релаксации."""
    A_t = transpose_matrix(A)
    if A == A_t and omega > 0.0 and omega < 2.0:
        return True
    else:
        return False
    
def krilov_method (A: list[list], c0: list[float]):
    """Вычисление собственного многочлена матрицы A методом Крылова."""
    n = len(A)
    c = []
    c.append(c0)

    for k in range (1, n+1):
        #print(k)
        ck = multiply_matrix(A, transpose(c[k-1]))
        c.append([ck[i][0] for i in range (n)])
        #print (f"c{k} (столбец): {c[k]}")

    temp_c = c[-(n-3)::-1]
    temp_c.append(c[-1])
    c = temp_c.copy()

    c_t = transpose_matrix(c)
    """
    print("Вектора C:")
    show_matrix(c)
    print("Трансп C:")
    show_matrix(c_t)
    """
    q, det = gaussian_method(c_t)
    return q

def get_proximity_measure (A: list[list]) -> float:
    """Вычисление меры близости матрицы A к диагональной."""
    proximity_measure = 0.0
    n = len(A)
    for i in range (n):
        for j in range (n):
            if i != j:
                proximity_measure += A[i][j]**2
    return proximity_measure

def find_max_off_diagonal_element(A: list[list]) -> tuple[int, int]:
    """Нахождение индексов максимального внедиагонального элемента симметричной матрицы A."""
    n = len(A)
    max_value = 0.0
    max_i = 0
    max_j = 0
    for i in range (n-1):
        for j in range (i+1, n):
            if i != j:
                if abs(A[i][j]) > abs(max_value):
                    max_value = A[i][j]
                    max_i = i
                    max_j = j
    return max_i, max_j

def rotation_matrix(size: int, i: int, j: int, cos_phi: float, sin_phi: float) -> list[list]:
    """Создание матрицы вращения размера size, которая поворачивает в плоскости (i, j) на угол phi."""
    T = [[0.0 if k != l else 1.0 for l in range(size)] for k in range(size)]
    T[i][i] = cos_phi
    T[j][j] = cos_phi
    T[i][j] = -sin_phi
    T[j][i] = sin_phi
    return T

def iterative_method_of_rotations(A: list[list], eps: float = 1e-5) -> tuple[list[float], list[list], int]:
    """Нахождение собственных значений и собственных векторов симметричной матрицы A методом вращений. Возвращает пару (вектор собственных значений, матрица собственных векторов, число итераций)."""
    n = len(A)
    eps = 1e-5
    t_A = get_proximity_measure(A)
    iter_count = 0

    U = [[1.0 if i==j else 0.0 for i in range(n)] for j in range(n)]

    while abs(t_A) > eps:
        i, j = find_max_off_diagonal_element(A)

        tg_2phi = 2*A[i][j]/(A[i][i]-A[j][j])
        cos_2phi = 1/( (1 + tg_2phi**2)**0.5 )
        cos_phi = ((1+cos_2phi)/2)**0.5 
        sin_phi = sign(A[i][j]*(A[i][i] - A[j][j]))*(((1-cos_2phi)/2)**0.5)

        B = [row[:] for row in A]
        
        for l in range (n): 
            B[l][i] = A[l][i]*cos_phi + A[l][j]*sin_phi
            B[l][j] = - A[l][i]*sin_phi + A[l][j]*cos_phi

        for l in range (n):
            A[i][l] = B[i][l]*cos_phi + B[j][l]*sin_phi
            A[j][l] = - B[i][l]*sin_phi + B[j][l]*cos_phi
            A[l][i] = A[i][l]
            A[l][j] = A[j][l]

        T = rotation_matrix(n, i, j, cos_phi, sin_phi)
        U = multiply_matrix(U, T)

        t_A = get_proximity_measure(A)
        iter_count += 1
        #print(f"Мера близости на итерации {iter_count}: {t_A}")

    eigenvalues = [A[i][i] for i in range (n)]
    eigenvectors = transpose_matrix(U)

    return eigenvalues, eigenvectors, iter_count

def simple_structure_matrix(A: list[list], k) -> list[list]:
    """Создание матрицы простой структуры размера len(A), в которой k-я строка отлична от единичной."""
    #на позиции k в числителе единица, знаменатель везде один
    n = len(A)
    M = [[0 if i!= j else 1 for j in range(n)] for i in range (n)]
    denominator = A[k+1][k]
    #print(f"denominator = A[{n-1}][{k}] = {denominator}")
    for i in range (n):
        M[k][i] = (-1)*A[k+1][i]/denominator
        #print (f"M[{k}][{i}] = A[{n-1}][{i}] / {denominator} = {M[k][i]}")
        if i == k:
            M[k][i] = 1/denominator
            #print (f"M[{k}][{i}] = 1 / {denominator} = {M[k][i]}")
    return M

def reversed_simple_structure_matrix(A: list[list], k) -> list[list]:
    """Создание обратной матрицы простой структуры размера len(A), в которой k-я строка отлична от единичной."""
    n = len(A)
    M_inv = [[0 if i!= j else 1 for j in range(n)] for i in range (n)]
    for i in range(n):
        M_inv[k][i] = A[k+1][i]
    return M_inv

def danilevski_method (A:list[list]) -> list[list]:
    """Нахождение матрицы Фробениуса F = S^-1*A*S методом Данилевского. Возвращает матрицу Фробениуса F и матрицу преобразования S."""
    F = [row[:] for row in A]
    n = len(A)

    S = [[1.0 if i==j else 0.0 for j in range(n)] for i in range (n)]

    for i in range (n-2, -1, -1):
        M_i = simple_structure_matrix(F, i)
        M_inv_i = reversed_simple_structure_matrix(F, i)

        S = multiply_matrix(S, M_i)
        """print(f"M_{i}:")
        show_matrix(S, 10, True)"""
        F = multiply_matrix(M_inv_i, F)
        F = multiply_matrix(F, M_i)
    return F, S

def power_method(A: list[list], eps: float = 1e-5) -> tuple[float, list[float], int]:
    """Нахождение наименьшего собственного значения и собственного вектора матрицы A методом степеней. Возвращает пару (собственное значение, собственный вектор, число итераций)."""
    n = len(A)
    y_prev = [1 for _ in range(n)]
    A_temp = [row[:] for row in A]
    for i in range (n):
        A_temp[i].append(y_prev[i])
    y_new, det = gaussian_method(A_temp)
    eigen_prev = 1.0
    eigen_new = multiply_vectors(y_new, y_prev)/multiply_vectors(y_prev, y_prev)
    norm = max(abs(v) for v in y_new)
    y_new = [v / norm for v in y_new]

    iter_count = 0
    while abs(eigen_new - eigen_prev) > eps:
        y_prev = y_new[:]
        eigen_prev = eigen_new
        A_temp = [row[:] for row in A]
        for i in range (n):
            A_temp[i].append(y_prev[i])
        y_new, det = gaussian_method(A_temp)
        eigen_new = multiply_vectors(y_new, y_prev)/multiply_vectors(y_prev, y_prev)
        norm = max(abs(v) for v in y_new)
        y_new = [v / norm for v in y_new]
        iter_count += 1
    
    min_eigenvalue = 1/eigen_new
    return min_eigenvalue, y_new, iter_count