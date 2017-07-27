from __future__ import print_function
import pickle
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

predictions = [ 2, 18,  0,  0, 28, 71, 27, 71,  6, 90, 14,  8, 35, 61, 27,  1,  6,  0,
  3,  0, 12, 71,  4, 27,  0,  0, 49,  0, 25, 49, 71,  3, 12,  5, 42, 39,
  3,  0, 40, 42,  0,  1,  0, 25, 38,  1, 28,  5,  0, 71, 15,  0,  8,  0,
 40,  0,  4, 25, 95,  0,  4,  1,  1, 71,  0,  0, 40, 71,  0, 71,  4, 36,
 39,  1, 28, 40, 40, 40, 18, 95,  0,  0, 40,  0, 28,  7, 71, 16, 40, 71,
  0, 71, 40, 28,  1, 71,  0, 71, 37, 40, 71, 71,  0, 22, 28, 40,  3, 35,
 42, 28,  0, 40, 40, 28, 71, 40,  0, 36, 17,  0, 37, 38, 39, 71,  0,  1,
 95,  2, 90, 71, 27, 40, 40, 71, 28, 40,  0,  0, 38,  2, 71,  0,  0, 71,
  3, 30,  0, 42, 71,  0,  3, 31, 19, 16, 91, 74,  0, 28, 21, 28, 38,  0,
 38, 28,  1, 40,  0,  0,  4, 20, 42, 28, 31,  0, 36, 95, 71,  0, 40, 39,
  0,  0,  0,  0, 40, 28, 40, 90, 49,  6, 39,  4, 71,  0, 37, 40,  2, 28,
  0,  8, 71, 40,  0,  4,  0, 61,  1,  0,  1, 27,  0, 40, 71, 28, 17,  0,
  0, 39, 23,  0,  4,  8,  0,  3,  3,  0, 90, 18, 71, 28, 39, 71,  0, 90,
 39,  0, 28, 32, 42,  0,  3,  8, 90,  1, 20, 90,  0,  1,  1,  0, 71, 40,
 18, 40, 28, 71,  4, 40,  0, 71, 95,  0, 71,  0, 40, 27, 49,  0, 25, 28,
 71, 25, 90,  0, 28, 71, 39, 34, 40,  0,  4,  0, 27,  0,  0,  0, 30, 22,
 39,  0, 40, 90,  0, 71, 49, 40,  0, 39,  0, 37,  9, 40,  0,  1,  4, 40,
  1, 40, 39,  0,  0, 28,  1, 40, 71, 12, 40,  0,  0, 71, 39,  6, 28,  1,
 16, 17, 35,  0,  1,  0,  0,  1, 38, 40,  2, 95, 28, 71, 39, 42, 25, 40,
  0,  0, 40, 17, 39, 16, 95,  3,  1, 10, 71,  1,  3,  3, 40, 39,  1, 39,
  0, 27, 95,  0,  0, 38, 90,  3,  0,  0, 23, 90, 39, 71, 71, 16,  0, 35,
 38, 39, 28, 90,  0, 95, 28,  0,  1,  0,  1, 95,  4, 28, 30,  3, 25, 95,
 22, 26, 90, 27, 23, 71,  0,  1, 91, 25, 25, 12, 17, 12, 71, 28, 71,  0,
  0,  1, 19,  5,  4, 71, 32, 40, 37, 40,  0,  0,  7,  1,  0,  4,  2, 28,
  0, 90,  7, 28, 27, 40, 90,  3,  7,  4, 19, 90, 39, 28, 14, 28,  0, 12,
  1, 71,  0, 28, 71, 28, 90, 31, 71,  0, 71,  0,  1, 28, 28,  1, 40,  0,
  0,  4, 27, 28, 42,  0,  0,  0,  0, 40,  0,  0,  0, 71, 27, 23,  3, 25,
 71, 16, 28, 31, 40, 39,  0, 95,  3,  2,  1,  2, 71,  0, 39,  0, 26, 40,
  0, 20, 39, 90,  6,  0,  0,  1,  0, 25,  3, 71, 37,  6,  8, 40, 71, 30,
 90,  0,  0, 10,  7,  4, 28, 40, 39, 74,  6, 71, 19,  0, 90, 10, 49,  0,
 40,  1,  0, 95,  3,  0,  9, 38, 30, 71,  3, 40, 28,  2, 42,  4,  0, 28,
  0, 28,  0, 15, 28,  0, 71,  7,  1, 28, 40,  1,  0, 95, 32,  1, 71,  0,
  4,  5, 37,  0, 71, 37,  0,  8, 17,  0, 90, 71,  0, 28,  2, 39, 38, 18,
  1, 40,  5,  7, 28,  0, 10, 95, 34,  2, 90, 71, 39, 28,  6,  0,  1, 27,
 28, 40,  2,  0,  3,  0,  0, 39, 27, 71,  1,  0, 42, 26, 40, 20, 71,  3,
 15,  1, 27, 71, 28,  0,  0, 90, 39,  1,  1,  0,  0,  1,  3, 33,  7,  1,
 20, 40,  1, 28,  1, 42,  0,  0, 71,  1,  7,  4, 39, 90, 28,  2,  0,  0,
  0,  0,  0, 71,  0, 39, 90, 28,  0, 90, 28, 28,  0, 39, 18,  3, 25, 39,
  0, 40, 37, 71, 39,  6, 71, 27, 40, 38, 40, 91, 20, 90,  0,  0, 40, 40,
 28, 71,  0, 90, 71, 71, 20, 40,  8,  4, 71,  0,  0,  0,  0,  0,  6, 39,
 71, 40,  0,  0, 90, 23, 19,  3,  0, 40, 95, 40,  0,  0,  1,  1, 40, 18,
 36,  0,  0, 71, 40,  0, 91, 71, 39, 71, 23, 71,  0, 71, 40,  4, 20, 12,
  0, 28,  0,  0, 28,  0, 71, 29,  9, 14,  0, 40, 27, 28,  1,  0,  2, 14,
  7, 71, 71, 35,  8, 28,  0, 28,  6,  1, 22,  0,  8, 28, 40, 38, 28, 40,
 28, 39,  0,  5,  0, 71,  7, 28, 30, 71,  0, 37, 40, 15,  0, 90, 40, 25,
 40, 28, 37, 90, 28,  0,  1, 22, 49, 23, 42, 71, 40, 39, 37, 28,  5,  0,
 28, 91,  4, 28, 40,  0, 90,  0, 35, 71,  0, 12, 28, 28, 28, 17, 91,  0,
 95,  0, 28, 71,  0, 17,  0, 10, 39,  0, 39, 90,  0,  4,  0, 36, 71, 49,
 71, 14, 39,  1, 71, 40, 90,  0,  1, 71, 71,  0,  7,  3,  9,  0, 61, 28,
 71, 10,  5, 28, 28,  4, 40,  0,  4, 40,  0, 12, 18, 28, 71,  3, 34,  0,
  4,  0, 33, 38,  0, 28,  1,  0, 91, 40, 38,  1, 40, 36,  4, 40, 71,  1,
 34, 22, 49, 25, 28, 95,  3,  0, 12, 26,  0, 28,  7,  1, 39, 28,  0,  0,
 10,  0, 27, 71,  1, 40, 28, 20,  1, 40,  1, 42,  0,  0,  1, 40, 17, 28,
  3, 39,  4,  0, 90,  1, 95, 40,  0,  0,  0, 71,  0, 71,  3,  1, 40, 90,
  1, 49,  7,  2,  9, 27, 38,  0,  0,  1,  0, 32,  0,  5, 40, 36, 32, 28,
 71, 40,  4,  0, 25, 42,  0, 40,  1, 28, 28, 90,  0,  8,  0, 28, 71, 40,
 77,  0, 38,  1,  4, 27,  0, 40, 40, 10,  0, 27,  0,  2, 90,  0,  0, 40,
 37, 74,  7, 25, 71, 39,  1, 35, 20,  0,  4,  0,  1,  1, 48,  4,  0,  0,
  0,  3, 71, 28,  7, 27, 28, 39,  0,  0, 18, 42,  4,  0, 11, 71, 28, 40,
 14, 95, 14,  0,  0,  4, 22,  0, 90,  0,  4,  6, 55, 42, 37, 10, 20,  0,
 28, 95,  8, 28, 16, 40, 27,  4,  5, 95, 49,  0, 39, 71, 90,  1, 16, 22,
 40,  0,  1, 40, 71, 12, 28,  0,  0, 40,  4,  3, 71, 14,  6,  0,  0,  0,
  3, 11, 40, 71,  0,  0, 49,  2,  8, 39, 28, 95, 71, 28, 42,  1, 71, 26,
  0, 95,  1, 95,  4, 19, 17, 40, 40,  1,  0, 25, 28,  0,  0, 16, 40, 71,
 49, 40, 27,  2,  1, 40,  3, 39, 27, 40, 28, 28, 39,  6, 28, 20,  4, 90,
 71, 36,  1, 38, 71,  1,  3,  0, 71,  0,  4, 71, 12,  0, 71, 71, 71, 10,
 26, 38, 71,  0,  8,  0,  4, 71, 39, 90, 42,  0, 95, 40,  0,  9,  0, 16,
 71,  0, 95,  0,  0, 14, 71, 14, 90, 28, 71, 42, 90,  0, 40, 71, 23, 18,
 95,  0,  2,  1,  6,  4,  5,  0, 40,  0,  1, 90, 40, 71,  1, 40,  0]

plt.hist(predictions, bins=101, normed=True)
plt.title('Distribution of Number of Cars Predicted Accross Train Samples (Cropped)')
plt.xlabel('Number of Cars Predicted to be in Photo')
plt.xlim(0,101)
plt.ylabel("Probability (Frequency / Total Number of Test Samples)")
plt.show()
