import numpy as np
import pickle

from lsh_forest import LSH_forest
from evaluation_functions import precision_test, precision_test_random_vector, precision_test_candidates_LSH_F, precision_test_candidates_LSH_F_random
from evaluation_functions import FLANN_funtions, ANNOY_functions
import matplotlib.pyplot as plt

"""
NOTE: You will need to create the data set and save as a pickle file as described in the following article:
http://maheshakya.github.io/gsoc/2014/05/18/preparing-a-bench-marking-data-set-using-singula-value-decomposition-on-movielens-data.html

Then save it in the working directory as '1M_k500_X.pickle'.
"""
X_lr = pickle.load(open('1M_k500_X.pickle', 'rb'))

#c values ranging from 10^(-1) to 10^3 in log scale with 100 samples
c_values= np.logspace(-1, 3, num = 100)

#Get precisions for LSH forest
precisions, times, candidates= precision_test_candidates_LSH_F(X_lr,limits=c_values)

#Get precisions for FLANN and ANNOY
limits = np.arange(10,1001,10)
flann_precisions, flann_times = precision_test(X_lr, limits=limits, imp=FLANN_funtions)
annoy_precisions, annoy_times = precision_test(X_lr, limits=limits, imp=ANNOY_functions)

#Plot precision vs time (of FLANN and LSH F)
x = np.array(times.values())/100.0
y = precisions.values()
plt.scatter(x, y, s=5, c='g')
x = np.array(flann_times.values())/100.0
y = flann_precisions.values()
plt.scatter(x, y, s=5, c='r')
plt.semilogx()
plt.ylabel('Precision percentage')
plt.xlabel('time')
plt.title("Precision vs. time")

plt.show()

#Plot precision vs time (of ANNOy and LSH forest)
x = np.array(times.values())/100.0
y = precisions.values()
plt.scatter(x, y, s=5, c='g')
x = np.array(annoy_times.values())/100.0
y = annoy_precisions.values()
plt.scatter(x, y, s=5, c='b')
plt.semilogx()
plt.ylabel('Precision percentage')
plt.xlabel('time')
plt.title("Precision vs. time")

plt.show()