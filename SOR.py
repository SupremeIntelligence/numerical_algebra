
from linear_algebra import input_matrix_from_file, sor_method, show_matrix, cube_norm, check_sor_convergence

A = input_matrix_from_file("input.txt")
b = []
eps = 1e-5
omega = 1.1
for row in A:
    b.append(row[-1])
    row.pop()
"""
omegas = [round(0.01 * k, 2) for k in range(100, 200, 10)]
results = []
for omega in omegas:
    out = sor_method(A, b, omega, init_x=b, eps=eps)
    if out[1] is None:
        iters = None
        rnorm = out[2]
    else:
        iters = out[1]
        rnorm = out[2]
    results.append((omega, iters, rnorm, out[0]))

# Print summary
print("omega    iterations    ||r||")
for omega, iters, r, sol in results:
    rnorm = cube_norm(r)
    iters_str = str(iters) if iters is not None else "no conv"
    print(f"{omega:6.2f}    {iters_str:10}    {rnorm:.2e}")

# find best omega (min iterations among converged)
converged = [row for row in results if row[1] is not None]
if converged:
    best = min(converged, key=lambda x: x[1])
    omega_best, iters_best, rnorm_best, sol_best = best
    print("\nBest omega:", omega_best, "iterations:", iters_best, "residual:", f"{rnorm_best:.2e}")
    print("Solution vector (best omega):")
    for i, val in enumerate(sol_best, start=1):
        print(f"x[{i}] = {val:.10f}")
else:
    print("\nNo omega in the tested grid converged within max_iter. Try other grid or check matrix.")
"""

if check_sor_convergence(A, omega):
    print("\n Метод верхней релаксации будет сходиться для данной системы.")
else:
    print("\nТеорема не выполняeтся. Сходимость не гарантируется.")

x, iter_count, r = sor_method(A, b, omega, b, eps)

print(f"\nРешение СЛАУ: x = {x}")
print(f"\nКоличество итераций: {iter_count}")
print("\nНевязка r:")
show_matrix(r, 10, True)
print(f"\nНорма r: {cube_norm(r)}")
