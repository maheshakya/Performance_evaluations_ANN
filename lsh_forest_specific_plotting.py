import numpy as np

from lsh_forest import LSH_forest
import matplotlib.pyplot as plt

from sklearn.metrics import euclidean_distances

"""
Create a dummy 2 dimensional data set for the visualization.
Create LSH forest with a single tree and build index with the dummy data.
"""
samples = 10000
dummy_x = np.random.rand(samples,2)
lshf = LSH_forest(number_of_trees=1)
lshf.build_index(dummy_x)

#Get candidate neighbors for a query
point = dummy_x[np.random.randint(0,samples)]
#point = np.random.rand(1,2)[0] #Use this if a random vector is required
neighbors, candidates = lshf.query_candidates(point, m=20)

#Plot candidate distribution with the query
x = dummy_x[[candidates],0]
y = dummy_x[[candidates],1]
plt.scatter(x, y, s=10, c='g')
plt.scatter(point[0], point[1], s=20, c='r')
plt.ylabel('Y')
plt.xlabel('X')
plt.title("Candidates distribution")
      
plt.show()

"""
For different values of length of hash, different number of candidates are returned.
As the number of hash bits increase, polytype of candidate distribution becomes more 
convex around the query point.
"""
number_of_hash_bits = 5
candidates = lshf.get_candidates_for_hash_length(point,number_of_hash_bits)

x = dummy_x[[candidates],0]
y = dummy_x[[candidates],1]
plt.scatter(x, y, s=10, c='g')
plt.scatter(point[0], point[1], s=40, c='r')
plt.ylabel('Y')
plt.xlabel('X')
plt.title("Data spread")
      
plt.show()

"""
Accuracy is tested by comparing with the exact nearest neighbors 
according to the Euclidean distance.
"""
accuracy = 0

for i in range(1000):
    point = dummy_x[np.random.randint(0,samples)]
    neighbors, candidates = lshf.query_candidates(point, c = 1, m=20)
    distances = euclidean_distances(point, dummy_x)
    ranks = np.argsort(distances)[0,:20]

    intersection = np.intersect1d(ranks, neighbors).shape[0]
    ratio = intersection/20.0    
    accuracy = accuracy + ratio
    print i, 'th iteration, accuracy: ', accuracy/(i+1)
    
print "Overral accuracy for a point in the data set: ", accuracy/1000.0
