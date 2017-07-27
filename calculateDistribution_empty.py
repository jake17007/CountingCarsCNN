from __future__ import print_function
import pickle
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# the data, shuffled and split between train and test sets
print('Importing data...')
x_train, y_train, x_test, y_test = pickle.load(open('feature_set_emptySpaces_clr.pickle', 'rb'))
print('Done importing data :)')

# combine train and test y's
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_combined = np.concatenate((y_train, y_test), axis=0)

def getDistribution(y_combined):
    targetCounts = []
    for target_j in range(len(y_combined[0])):
        sumOfTarget_j = 0
        for y_i in y_combined:
            if y_i[target_j] == 1:
                sumOfTarget_j += 1
        targetCounts.append(sumOfTarget_j)
    return targetCounts

def convertFromOneHotToNumerical(y_combined):
    numericalTargets = []
    for y_i in y_combined:
        for j in range(len(y_i)):
            if y_i[j] == 1:
                numericalTargets.append(j)
    return numericalTargets


numericalTargets = convertFromOneHotToNumerical(y_combined)

plt.hist(numericalTargets, bins=20, normed=True)
plt.title('Distribution of Empty Parking Spaces Accross Samples (Cropped)')
plt.xlabel('Number of Empty Parking Spaces in Photo')
plt.xlim(0,19)
plt.ylabel("Probability (Frequency / Total Number of Samples)")
plt.show()
