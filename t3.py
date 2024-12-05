import numpy as np


def jacobi(A, b, x0, num_iter):
    n = len(A)
    x = x0.copy()
    for iteration in range(num_iter):
        x_new = np.array([(b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i] for i in range(n)])
        print(f"Jacobi迭代第{iteration + 1}次结果: {x_new}")
        x = x_new
    return x


def gauss_seidel(A, b, x0, num_iter):
    n = len(A)
    x = x0.copy()
    for iteration in range(num_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        print(f"Gauss - Seidel迭代第{iteration + 1}次结果: {x_new}")
        x = x_new
    return x


# # 定义系数矩阵A
# def create_tridiagonal_matrix_and_b(n):
#     diagonal = np.array([3 if i == 0 or i == n - 1 else 2 for i in range(n)])
#     A = np.diag(diagonal) + np.diag([-1] * (n - 1), 1) + np.diag([-1] * (n - 1), -1)

#     b = np.zeros(n)
#     for i in range(n):
#         b[i] = 1.0
#     return A, b


# n = 3

# 调用函数创建三对角矩阵A和常数向量b
# A, b = create_tridiagonal_matrix_and_b(n)

# 初始解向量
# x0 = np.zeros(n)
# 定义系数矩阵A
A = np.array([[10, -2, -1],
              [-2, 10, -1],
              [-1, -2, 5]])
# 定义常数向量b
b = np.array([3, 15, 10])
# 初始解向量
x0 = np.zeros(3)
# 获取用户输入的迭代步数
num_iterations = 10
# print("生成的三对角矩阵A为：")
# print(A)
# print("生成的三对角矩阵b为：")
# print(b)


print("Jacobi迭代法求解结果:")
solution_jacobi = jacobi(A, b, x0, num_iterations)
print("最终结果:", solution_jacobi)

print("\nGauss - Seidel迭代法求解结果:")
solution_gauss_seidel = gauss_seidel(A, b, x0, num_iterations)
print("最终结果:", solution_gauss_seidel)

print("\n两种方法最终结果对比:")
print("Jacobi迭代法:", solution_jacobi)
print("Gauss - Seidel迭代法:", solution_gauss_seidel)