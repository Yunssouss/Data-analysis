import numpy as np

# نصاوب مصفوفة
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# عملية جمع المصفوفات
matrix_sum = matrix1 + matrix2

# عملية الضرب
matrix_product = np.dot(matrix1, matrix2)

print("Sum of matrices:\n", matrix_sum)
print("Product of matrices:\n", matrix_product)

