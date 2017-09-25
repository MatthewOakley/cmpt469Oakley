# question3.py
# Written by Matthew Oakley
# 9/11/17
# This program gets the rank to two matrices
# that are multiplied together

import numpy as np

a = np.matrix('1 4 -3; 2 -1 3')
print("Matrix A:")
print(a)

b = np.matrix('-2 0 5; 0 -1 4')
print("Matrix B:")
print(b)

# A and B cannot be multiplied causes an error
#print(a.dot(b))

print("Transposed A:")
c = (a.transpose())
print(c)

print("Matrix A * Matrix B:")
print(c.dot(b))

# gives weird error but still works
print(np.rank((c.dot(b))))