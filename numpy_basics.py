import numpy as np


a = [9, 3, 3, 5]
arr = np.array(a)
print("Array:", arr)


print("Shape:", arr.shape)
print("Size:", arr.size)
print("Data Type:", arr.dtype)


print("First Element:", arr[0])
print("Last Element:", arr[-1])


print("Slice:", arr[1:3])


print("Sum:", arr.sum())
print("Mean:", arr.mean())
print("Maximum:", arr.max())
print("Minimum:", arr.min())


marks = np.array([75, 82, 90, 68, 95])

print("Total Marks:", marks.sum())
print("Average Marks:", marks.mean())
print("Highest Mark:", marks.max())
print("Lowest Mark:", marks.min())