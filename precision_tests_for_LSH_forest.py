import numpy as np
import pickle

from lsh_forest import LSH_forest
from evaluation_functions import precision_test, precision_test_random_vector, precision_test_candidates_LSH_F, precision_test_candidates_LSH_F_random
import matplotlib.pyplot as plt

"""
NOTE: You will need to create the data set and save as a pickle file as described in the following article:
http://maheshakya.github.io/gsoc/2014/05/18/preparing-a-bench-marking-data-set-using-singula-value-decomposition-on-movielens-data.html

Then save it in the working directory as '1M_k500_X.pickle'.
"""
X_lr = pickle.load(open('1M_k500_X.pickle', 'rb'))

#c values ranging from 10^(-1) to 10^3 in log scale with 100 samples
c_values= np.logspace(-1, 3, num = 100)

precisions, times, candidates= precision_test_candidates_LSH_F(X_lr,limits=c_values)

#Plot precision vs c
x = precisions.keys()
y = precisions.values()
plt.scatter(x, y, s=5, c='g')
plt.semilogx()
plt.ylabel('Precision percentage')
plt.xlabel('c')
plt.title("Precision vs. c")

plt.show()

#Plot precision vs time
x = np.array(times.values())/100.0
y = precisions.values()
plt.scatter(x, y, s=5, c='g')
plt.semilogx()
plt.ylabel('Precision percentage')
plt.xlabel('Time in seconds')
plt.title("Precision vs. time")

plt.show()

#Plot precision vs times
x = np.array(candidates.values())/100.0
y = precisions.values()
plt.scatter(x, y, s=5, c='g')
plt.semilogx()
plt.ylabel('Precision percentage')
plt.xlabel('number of candidates')
plt.title("Precision vs. number of candidates")

plt.show()

x = candidates.keys()
y = np.array(candidates.values())/100.0
plt.scatter(x, y, s=5, c='g')
plt.semilogx()
plt.ylabel('Number of candidates')
plt.xlabel('c')
plt.title("Number of candidates vs. c")

plt.show()

