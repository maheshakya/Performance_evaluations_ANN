import numpy as np
import scipy as sp
import pickle, time 

from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
from pykgraph import KGraph
from pyflann import FLANN
from lshash import LSHash
from lsh_forest import LSH_forest

import nearpy
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

#sklearn class
class sklearn_funtions():
    def build_index(self, X):
        nbrs = NearestNeighbors()
        nbrs.fit(X)
        return nbrs
    
    def query(self, X, t, query_vector, number_of_neighbors):
        distances, indices = t.kneighbors(X[query_vector], n_neighbors=number_of_neighbors)
        return list(indices[0])
    
    def query_random_vector(self, t, query_vector, number_of_neighbors):
        distances, indices = t.kneighbors(query_vector, n_neighbors=number_of_neighbors)
        return list(indices[0])
    
    def query_params(self, X, index_structure, query_vector, number_of_neighbors):
        params = {'X':X, 't':index_structure, 'query_vector':query_vector, 'number_of_neighbors':number_of_neighbors}
        return params
        
#ANNOY class
class ANNOY_functions():
    def build_index(self,X):
        f = X.shape[1]
        n = X.shape[0]

        t = AnnoyIndex(f, metric = 'angular')

        for i in range(n):
            t.add_item(i, X[i].tolist())

        t.build(-1)   
        return t

    def query(self, t, query_vector, number_of_neighbors):
        return t.get_nns_by_item(query_vector, number_of_neighbors)

    def query_random_vector(self, t, query_vector, number_of_neighbors):
        return t.get_nns_by_vector(query_vector.tolist(), number_of_neighbors)

    def query_params(self, X, index_structure, query_vector, number_of_neighbors):
        params = {'t':index_structure, 'query_vector':query_vector, 'number_of_neighbors':number_of_neighbors}
        return params

#KGraph class
class KGraph_functions():
    def build_index(self, X):
        index = KGraph()
        index.build(X)
        return index

    def query(self, X, t, query_vector, number_of_neighbors):
        return t.search(data = X, query = X[query_vector:query_vector+1], K = number_of_neighbors)[0]

    def query_random_vector(self, X, t, query_vector, number_of_neighbors):
        return t.search(data = X, query = np.array([query_vector]), K = number_of_neighbors)[0]

    def query_params(self, X, index_structure, query_vector, number_of_neighbors):
        params = {'X':X, 't':index_structure, 'query_vector':query_vector, 'number_of_neighbors':number_of_neighbors}
        return params

#FLANN class
class FLANN_funtions(): 
    def build_index(self, X):
        flann = FLANN()
        params = flann.build_index(X)

        return flann

    def query(self, X, t, query_vector, number_of_neighbors):
        neighbors, dists  = t.nn_index(X[query_vector], number_of_neighbors)
        return neighbors[0]

    def query_random_vector(self, X, t, query_vector, number_of_neighbors):
        neighbors, dists  = t.nn_index(query_vector, number_of_neighbors)
        return neighbors[0]

    def query_params(self, X, index_structure, query_vector, number_of_neighbors):
        params = {'X':X, 't':index_structure, 'query_vector':query_vector, 'number_of_neighbors':number_of_neighbors}
        return params

#nearpy class
class nearpy_funtions():
    def build_index(self, X):
        f = X.shape[1]
        n = X.shape[0]

        rbp = RandomBinaryProjections('rbp', 32)
        engine = Engine(f, lshashes=[rbp])

        for i in range(n):
            engine.store_vector(X[i], 'data_%d' % i)

        return engine

    def query(sefl, X, t, query_vector):    
        return t.neighbours(X[query_vector])

    def query_random_vector(self, X, t, query_vector):    
        return t.neighbours(query_vector)

    def query_params(self, X, index_structure, query_vector, number_of_neighbors):
        params = {'X':X, 't': index_structure, 'query_vector':query_vector}
        return params

#lshash class
class lshash_functions():
    def build_index(self, X):
        f = X.shape[1]
        n = X.shape[0]

        lsh = LSHash(hash_size = 32, input_dim = f, num_hashtables = 100)
        for i in range(n):
            lsh.index(X[i], i)

        return lsh

    def query(self, X, t, query_vector):    
        return t.query(query_point = X[query_vector])

    def query_random_vector(self, X, t, query_vector):    
        return t.query(query_point = query_vector)

    def query_params(self, X, index_structure, query_vector, number_of_neighbors):
        params = {'X':X, 't': index_structure, 'query_vector':query_vector}
        return params

#lsh_forest class
class lsh_forest_functions():
    def build_index(self, X):
        f = X.shape[1]
        n = X.shape[0]
        
        lsh_forest = LSH_forest(number_of_trees=10)
        lsh_forest.build_index(X)
        
        return lsh_forest
    
    def query(self, X, t, query_vector, c):
        return t.query(X[query_vector], c=c)
    
    def query_random_vector(self, X, t, query_vector, c):
        return t.query(query_vector, c=c)
    
    def query_params(self, X, index_structure, query_vector, c):
        params = {'X':X, 't':index_structure, 'query_vector':query_vector, 'c':c}
        return params
        
"""
Query speed tests for nearest neighbor
"""

#Qyery speed tests for nearest neighbor : data point in the data set
def queryTimes_nearest(X, imp = ANNOY_functions):
    f = X.shape[1]
    n = X.shape[0]
    
    ann = imp()
    t = ann.build_index(X) 

    lim_n = 1000
    time_sum = 0

    for i in xrange(lim_n):
        j = np.random.randint(0, n)
        print 'finding the nearest neighnor for', j        
        params = ann.query_params(X, t, j, 1)
        
        t0 = time.time()
        neighbors = ann.query(**params)
        T = time.time() - t0
        
        time_sum = time_sum + T
        
        print 'avg time: %.6fs' % (time_sum / (i + 1))
        
    return time_sum / lim_n
        
#Qyery speed tests for nearest neighbor : a random data point
def queryTimes_nearest_random_vector(X, imp = ANNOY_functions):
    f = X.shape[1]
    n = X.shape[0]
    
    ann = imp()
    t = ann.build_index(X)

    lim_n = 1000
    time_sum = 0

    for i in xrange(lim_n):
        query_vector = np.random.random(size=f)
        print 'finding the nearest neighnor for a random vector'      
        params = ann.query_params(X, t, query_vector, 1)
        
        t0 = time.time()
        neighbors = ann.query(**params)
        T = time.time() - t0
        
        time_sum = time_sum + T
        
        print 'avg time: %.6fs' % (time_sum / (i + 1))
        
    return time_sum / lim_n
            

"""
Query speed tests for multiple nearest neighbors
"""

# a data point in the data set
def queryTimes_multiple(X, limits = [1, 10, 100, 1000] , imp = ANNOY_functions, iterations = 100):
    f = X.shape[1]
    n = X.shape[0]
    
    ann = imp()
    t = ann.build_index(X)
    
    limits = limits
    lim_n = iterations
    time_sum = {}

    for i in xrange(lim_n):
        j = np.random.randint(0, n)
        print 'finding nbs for', j        
        
        for limit in limits:
            params = ann.query_params(X, t, j, limit)
            t0 = time.time()
            neighbors = ann.query(**params)
            T = time.time() - t0

            time_sum[limit] = time_sum.get(limit, 0.0) + T

        for limit in limits:
            print 'limit: %-9d avg time: %.6fs' % (limit, time_sum[limit] / (i + 1))
            
    return time_sum

# a random data point
def queryTimes_multiple_random_vector(X, limits = [1, 10, 100, 1000] , imp = ANNOY_functions, iterations = 100):
    f = X.shape[1]
    n = X.shape[0]
    
    ann = imp()
    t = ann.build_index(X)
    
    limits = limits
    lim_n = iterations
    time_sum = {}

    for i in xrange(lim_n):
        query_vector = np.random.random(size=f)
        print 'finding the nearest neighnor for a random vector'        
        
        for limit in limits:
            params = ann.query_params(X, t, query_vector, limit)
            t0 = time.time()
            neighbors = ann.query(**params)
            T = time.time() - t0

            time_sum[limit] = time_sum.get(limit, 0.0) + T

        for limit in limits:
            print 'limit: %-9d avg time: %.6fs' % (limit, time_sum[limit] / (i + 1))
            
    return time_sum

"""
Precision tests for ANNOY, FLANN, KGraph
"""

# a data point in the data set
def precision_test(X, limits = [10, 100, 1000] , imp = ANNOY_functions, iterations = 100):
    f = X.shape[1]
    n = X.shape[0]
    
    ann = imp()
    t = ann.build_index(X)
    
    limits = limits
    k = 10
    prec_sum = {}
    prec_n = iterations
    time_sum = {}

    for i in xrange(prec_n):
        j = np.random.randint(0, n)
        print 'finding nbs for', j
        params = ann.query_params(X, t, j, n)
        
        closest = set(ann.query(**params)[:k])
        for limit in limits:
            params = ann.query_params(X, t, j, limit)
            t0 = time.time()
            toplist = ann.query(**params)
            T = time.time() - t0
            
            found = len(closest.intersection(toplist))
            hitrate = 1.0 * found / k
            prec_sum[limit] = prec_sum.get(limit, 0.0) + hitrate
            time_sum[limit] = time_sum.get(limit, 0.0) + T

        for limit in limits:
            print 'limit: %-9d precision: %6.2f%% avg time: %.6fs' % (limit, 100.0 * 
                                                                      prec_sum[limit] / (i + 1), time_sum[limit] / (i + 1))
    
    return prec_sum, time_sum

# a random data point
def precision_test_random_vector(X, limits = [10, 100, 1000] , imp = ANNOY_functions, iterations = 100):
    f = X.shape[1]
    n = X.shape[0]
    
    ann = imp()
    t = ann.build_index(X)  
    
    limits = limits
    k = 10
    prec_sum = {}
    prec_n = iterations
    time_sum = {}

    for i in xrange(prec_n):
        query_vector = np.random.random(size=f)
        print 'finding the nearest neighnor for a random vector'
        params = ann.query_params(X, t, query_vector, n)
        
        closest = set(ann.query_random_vector(**params)[:k])
        for limit in limits:
            params = ann.query_params(X, t, query_vector, limit)
            t0 = time.time()
            toplist = ann.query_random_vector(**params)
            T = time.time() - t0
            
            found = len(closest.intersection(toplist))
            hitrate = 1.0 * found / k
            prec_sum[limit] = prec_sum.get(limit, 0.0) + hitrate
            time_sum[limit] = time_sum.get(limit, 0.0) + T

        for limit in limits:
            print 'limit: %-9d precision: %6.2f%% avg time: %.6fs' % (limit, 100.0 * 
                                                                      prec_sum[limit] / (i + 1), time_sum[limit] / (i + 1))
    
    return prec_sum, time_sum


"""
Precision test for lsh_forest
"""
# a data point in the data set
def precision_test_candidates_LSH_F(X, limits = [10, 100, 1000], iterations = 100):
    f = X.shape[1]
    n = X.shape[0]

    t = LSH_forest(number_of_trees=10)
    t.build_index(X)
    
    limits = limits
    k = 10
    prec_sum = {}
    prec_n = iterations
    time_sum = {}
    candidate_sum = {}

    for i in xrange(prec_n):
        j = np.random.randint(0, n)
        print 'finding nbs for', j
        
        neighbors, candidates = t.query_num_candidates(X[j], c=n)
        closest = set(neighbors[:k])
        for limit in limits:
            t0 = time.time()
            neighbors, candidates= t.query_num_candidates(X[j], c=limit)
            T = time.time() - t0
            toplist = neighbors[:k]
            
            found = len(closest.intersection(toplist))
            hitrate = 1.0 * found / k
            prec_sum[limit] = prec_sum.get(limit, 0.0) + hitrate
            time_sum[limit] = time_sum.get(limit, 0.0) + T
            candidate_sum[limit] = candidate_sum.get(limit, 0) + candidates            

        for limit in limits:
            print 'limit: %.8f precision: %6.2f%% avg time: %.6fs avg_candiates: %.1f' % (limit, 100.0 * 
                                                                      prec_sum[limit] / (i + 1), time_sum[limit] / (i + 1), 
                                                                      candidate_sum[limit]/ (i+1))
    
    return prec_sum, time_sum, candidate_sum


# random vector
def precision_test_candidates_LSH_F_random(X, limits = [10, 100, 1000], iterations = 100):
    f = X.shape[1]
    n = X.shape[0]

    t = LSH_forest(number_of_trees=10)
    t.build_index(X)
    
    limits = limits
    k = 10
    prec_sum = {}
    prec_n = iterations
    time_sum = {}
    candidate_sum = {}

    for i in xrange(prec_n):
        query_vector = np.random.random(size=f)
        print 'finding nbs for a random vector'
        
        neighbors, candidates = t.query_num_candidates(query_vector, c=n)
        closest = set(neighbors[:k])
        for limit in limits:
            t0 = time.time()
            neighbors, candidates= t.query_num_candidates(query_vector, c=limit)
            T = time.time() - t0
            toplist = neighbors[:k]
            
            found = len(closest.intersection(toplist))
            hitrate = 1.0 * found / k
            prec_sum[limit] = prec_sum.get(limit, 0.0) + hitrate
            time_sum[limit] = time_sum.get(limit, 0.0) + T
            candidate_sum[limit] = candidate_sum.get(limit, 0) + candidates            

        for limit in limits:
            print 'limit: %.8f precision: %6.2f%% avg time: %.6fs avg_candiates: %.1f' % (limit, 100.0 * 
                                                                      prec_sum[limit] / (i + 1), time_sum[limit] / (i + 1), 
                                                                      candidate_sum[limit]/ (i+1))
    
    return prec_sum, time_sum, candidate_sum

