import numpy as np


def jacobi(A, b, x0, max_iter=1000, tol=1e-6):
    """
    Jacobi迭代法求解线性方程组Ax = b，并输出每一步迭代结果
    """
    n = len(A)
    x = x0.copy()
    for iteration in range(max_iter):
        x_new = np.array([(b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i] for i in range(n)])
        print(f"Jacobi迭代第{iteration + 1}次结果: {x_new}")
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x


def gauss_seidel(A, b, x0, max_iter=1000, tol=1e-6):
    """
    Gauss - Seidel迭代法求解线性方程组Ax = b，并输出每一步迭代结果
    """
    n = len(A)
    x = x0.copy()
    for iteration in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        print(f"Gauss - Seidel迭代第{iteration + 1}次结果: {x_new}")
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x


# 定义系数矩阵A
A = np.array([[3, -1, 1],
              [-1, 3, -1],
              [1, -1, 3]])
# 定义常数向量b
b = np.array([1, 1, 1])
# 初始解向量
x0 = np.zeros(3)

print("Jacobi迭代法求解结果:")
solution_jacobi = jacobi(A, b, x0)
print("最终方程组的解:", solution_jacobi)

print("\nGauss - Seidel迭代法求解结果:")
solution_gauss_seidel = gauss_seidel(A, b, x0)
print("最终方程组的解:", solution_gauss_seidel)