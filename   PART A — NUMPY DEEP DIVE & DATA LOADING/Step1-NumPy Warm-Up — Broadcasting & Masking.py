# Step 1 — NumPy Warm-Up: Broadcasting & Masking

## Objective
#The goal of this task is to practice NumPy fundamentals including broadcasting, normalization, masking, conditional replacement, and dot product operations.

#---

## (a) Create Random Score Matrix
#We generated a 6×4 matrix representing scores of 6 students across 4 subjects using NumPy’s random integer function.

#---

## (b) Column-wise Normalization
#Each column was normalized to the range [0,1] using broadcasting:

#Normalized Formula:
#(X - min) / (max - min)

#This ensures all values are scaled consistently without using loops.

#---

## (c) Boolean Masking (Passing Students)
#We computed the average score per student and created a boolean mask for students scoring ≥ 60. Using this mask, we extracted only the passing students.

#---

## (d) Conditional Replacement using np.where
#Scores below 40 were replaced with the column mean using NumPy’s vectorized `np.where` function.

#---

## (e) Weighted Scoring using Dot Product
#A random weight vector was generated and multiplied with the normalized matrix using dot product to simulate a weighted scoring system.

#---

## Conclusion
#This exercise demonstrated:
#- Efficient array operations using broadcasting
#- Data filtering using boolean masks
#- Conditional transformations with np.where
#- Linear algebra operations with dot product

#These concepts are essential for data preprocessing and machine learning workflows.

import numpy as np

# (a) Create a 6x4 matrix of random integers (0–100)
np.random.seed(42)  # for reproducibility
scores = np.random.randint(0, 101, size=(6, 4))

print("Original Scores (6 students × 4 exams):\n", scores)

# --------------------------------------------------

# (b) Normalize each column to [0,1] using broadcasting
col_min = scores.min(axis=0)
col_max = scores.max(axis=0)

normalized = (scores - col_min) / (col_max - col_min)

print("\nColumn-wise Min:\n", col_min)
print("Column-wise Max:\n", col_max)
print("\nNormalized Scores:\n", normalized)

# --------------------------------------------------

# (c) Boolean mask for students averaging >= 60
averages = scores.mean(axis=1)
mask = averages >= 60

passing_students = scores[mask]

print("\nStudent Averages:\n", averages)
print("Passing Mask (>=60):\n", mask)
print("\nPassing Students:\n", passing_students)

# --------------------------------------------------

# (d) Replace scores below 40 with column mean using np.where
col_mean = scores.mean(axis=0)

adjusted_scores = np.where(scores < 40, col_mean, scores)

print("\nColumn Means:\n", col_mean)
print("\nAdjusted Scores (values < 40 replaced):\n", adjusted_scores)

# --------------------------------------------------

# (e) Dot product with random weight vector
weights = np.random.rand(4)

weighted_scores = normalized.dot(weights)

print("\nRandom Weights:\n", weights)
print("\nWeighted Scores (Dot Product Result):\n", weighted_scores)