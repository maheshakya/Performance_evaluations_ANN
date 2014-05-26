import numpy as np
import scipy as sp
import pickle, time 

from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
from pykgraph import KGraph
from pyflann import FLANN
from lshash import LSHash

import nearpy
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

##Functions for kgraph
def build_index_kgraph(X):
    index = KGraph()
    index.build(X)
    return index

##Functions for ANNOY
def build_index_ANNOY(X):
    f = X.shape[1]
    n = X.shape[0]
    
    t = AnnoyIndex(f, metric = 'angular')
            
    for i in range(n):
        t.add_item(i, X[i].tolist())

    t.build(-1)   
    return t

def query_ANNOY(t, item_num, number_of_neighbors):
    return t.get_nns_by_item(item_num, number_of_neighbors)

def query_ANNOY_random_vector(t, query_vector, number_of_neighbors):
    return t.get_nns_by_vector(query_vector.tolist(), number_of_neighbors)

def query_params_ANNOY(X, index_structure, item_num, number_of_neighbors):
    params = {'t':index_structure, 'item_num':item_num, 'number_of_neighbors':number_of_neighbors}
    return params

def query_params_ANNOY_random_vector(X, index_structure, query_vector, number_of_neighbors):
    params = {'t':index_structure, 'query_vector':query_vector, 'number_of_neighbors':number_of_neighbors}
    return params

##Functions for FLANN
def build_index_FLANN(X):
    flann = FLANN()
    params = flann.build_index(X)
    
    return flann

def query_FLANN(X, flann, item_num, number_of_neighbors):
    neighbors, dists  = flann.nn_index(X[item_num], number_of_neighbors)
    return neighbors[0]

def query_FLANN_random_vector(X, flann, query_vector, number_of_neighbors):
    neighbors, dists  = flann.nn_index(query_vector, number_of_neighbors)
    return neighbors[0]

def query_params_FLANN(X, index_structure, item_num, number_of_neighbors):
    params = {'X':X, 'flann':index_structure, 'item_num':item_num, 'number_of_neighbors':number_of_neighbors}
    return params

def query_params_FLANN_random_vector(X, index_structure, query_vector, number_of_neighbors):
    params = {'X':X, 'flann':index_structure, 'query_vector':query_vector, 'number_of_neighbors':number_of_neighbors}
    return params

##Functions for nearpy
def build_index_nearpy(X):
    f = X.shape[1]
    n = X.shape[0]
    
    rbp = RandomBinaryProjections('rbp', 32)
    engine = Engine(f, lshashes=[rbp])

    for i in range(n):
        engine.store_vector(X[i], 'data_%d' % i)
        
    return engine

def query_nearpy(X, engine, item_num):    
    return engine.neighbours(X[item_num])

def query_nearpy_random_vector(X, engine, query_vector):    
    return engine.neighbours(query_vector)

def query_params_nearpy(X, index_structure, item_num, number_of_neighbors):
    params = {'X':X, 'engine': index_structure, 'item_num':item_num}
    return params
    
def query_params_nearpy_random_vector(X, index_structure, query_vector, number_of_neighbors):
    params = {'X':X, 'engine': index_structure, 'query_vector':query_vector}
    return params

##Functions for lshash
def build_index_lshash(X):
    f = X.shape[1]
    n = X.shape[0]
    
    lsh = LSHash(hash_size = 32, input_dim = f, num_hashtables = 100)
    for i in range(n):
        lsh.index(X[i], i)
        
    return lsh

def query_lshash(X, lsh, item_num):    
    return lsh.query(query_point = X[item_num])

def query_lshash_random_vector(X, lsh, query_vector):    
    return lsh.query(query_point = query_vector)

def query_params_lshash(X, index_structure, item_num, number_of_neighbors):
    params = {'X':X, 'lsh': index_structure, 'item_num':item_num}
    return params

def query_params_lshash_random_vector(X, index_structure, query_vector, number_of_neighbors):
    params = {'X':X, 'lsh': index_structure, 'query_vector':query_vector}
    return params

""" Query speed tests for nearest neighbor """

#Qyery speed tests for nearest neighbor : data point in the data set
def queryTimes_nearest(X, query_params = query_params_ANNOY, indexing = build_index_ANNOY, query = query_ANNOY):
    f = X.shape[1]
    n = X.shape[0]

    t = indexing(X) 

    lim_n = 1000
    time_sum = 0

    for i in xrange(lim_n):
        j = np.random.randint(0, n)
        print 'finding the nearest neighnor for', j        
        params = query_params(X, t, j, 1)
        
        t0 = time.time()
        neighbors = query(**params)
        T = time.time() - t0
        
        time_sum = time_sum + T
        
        print 'avg time: %.6fs' % (time_sum / (i + 1))
        
    return time_sum / lim_n

#Qyery speed tests for nearest neighbor : a random data point
def queryTimes_nearest_random_vector(X, query_params = query_params_ANNOY_random_vector, 
                                     indexing = build_index_ANNOY, query = query_ANNOY_random_vector):
    f = X.shape[1]
    n = X.shape[0]

    t = indexing(X) 

    lim_n = 1000
    time_sum = 0

    for i in xrange(lim_n):
        query_vector = np.random.random(size=f)
        print 'finding the nearest neighnor for a random vector'      
        params = query_params(X, t, query_vector, 1)
        
        t0 = time.time()
        neighbors = query(**params)
        T = time.time() - t0
        
        time_sum = time_sum + T
        
        print 'avg time: %.6fs' % (time_sum / (i + 1))
        
    return time_sum / lim_n

""" Query speed tests for multiple nearest neighbors """
            
# a data point in the data set
def queryTimes_multiple(X, query_params = query_params_ANNOY, indexing = build_index_ANNOY, query = query_ANNOY):
    f = X.shape[1]
    n = X.shape[0]
    
    t = indexing(X)     
    
    limits = np.arange(1, 6002, 500)
    lim_n = 100
    time_sum = {}

    for i in xrange(lim_n):
        j = np.random.randint(0, n)
        print 'finding nbs for', j        
        
        for limit in limits:
            params = query_params(X, t, j, limit)
            t0 = time.time()
            neighbors = query(**params)
            T = time.time() - t0

            time_sum[limit] = time_sum.get(limit, 0.0) + T

        for limit in limits:
            print 'limit: %-9d avg time: %.6fs' % (limit, time_sum[limit] / (i + 1))
            
    return limits, time_sum, lim_n  
            

# a random data point
def queryTimes_multiple_random_vector(X, query_params = query_params_ANNOY_random_vector, indexing 
                        = build_index_ANNOY, query = query_ANNOY_random_vector):
    f = X.shape[1]
    n = X.shape[0]
    
    t = indexing(X)     
    
    limits = np.arange(1, 6002, 500)
    lim_n = 100
    time_sum = {}

    for i in xrange(lim_n):
        query_vector = np.random.random(size=f)
        print 'finding the nearest neighnor for a random vector'        
        
        for limit in limits:
            params = query_params(X, t, query_vector, limit)
            t0 = time.time()
            neighbors = query(**params)
            T = time.time() - t0

            time_sum[limit] = time_sum.get(limit, 0.0) + T

        for limit in limits:
            print 'limit: %-9d avg time: %.6fs' % (limit, time_sum[limit] / (i + 1))
            
    return limits, time_sum, lim_n

""" Precision tests for ANNOY and FLANN """

# a data point in the data set
def precision_test(X, query_params = query_params_ANNOY, indexing = build_index_ANNOY, query = query_ANNOY):
    f = X.shape[1]
    n = X.shape[0]
    
    t = indexing(X)  
    
    limits = [10, 100, 1000]
    print limits
    k = 10
    prec_sum = {}
    prec_n = 100
    time_sum = {}

    for i in xrange(prec_n):
        j = np.random.randint(0, n)
        print 'finding nbs for', j
        params = query_params(X, t, j, n)
        
        closest = set(query(**params)[:k])
        for limit in limits:
            params = query_params(X, t, j, limit)
            t0 = time.time()
            toplist = query(**params)
            T = time.time() - t0
            
            found = len(closest.intersection(toplist))
            hitrate = 1.0 * found / k
            prec_sum[limit] = prec_sum.get(limit, 0.0) + hitrate
            time_sum[limit] = time_sum.get(limit, 0.0) + T

        for limit in limits:
            print 'limit: %-9d precision: %6.2f%% avg time: %.6fs' % (limit, 100.0 * 
                                                                      prec_sum[limit] / (i + 1), time_sum[limit] / (i + 1))
    
    return limits, prec_sum, time_sum, prec_n

# a random data point

def precision_test_random_vector(X, query_params = query_params_ANNOY_random_vector, 
                                 indexing = build_index_ANNOY, query = query_ANNOY_random_vector):
    f = X.shape[1]
    n = X.shape[0]
    
    t = indexing(X)  
    
    limits = [10, 100, 1000]
    print limits
    k = 10
    prec_sum = {}
    prec_n = 100
    time_sum = {}

    for i in xrange(prec_n):
        query_vector = np.random.random(size=f)
        print 'finding the nearest neighnor for a random vector'
        params = query_params(X, t, query_vector, n)
        
        closest = set(query(**params)[:k])
        for limit in limits:
            params = query_params(X, t, query_vector, limit)
            t0 = time.time()
            toplist = query(**params)
            T = time.time() - t0
            
            found = len(closest.intersection(toplist))
            hitrate = 1.0 * found / k
            prec_sum[limit] = prec_sum.get(limit, 0.0) + hitrate
            time_sum[limit] = time_sum.get(limit, 0.0) + T

        for limit in limits:
            print 'limit: %-9d precision: %6.2f%% avg time: %.6fs' % (limit, 100.0 * 
                                                                      prec_sum[limit] / (i + 1), time_sum[limit] / (i + 1))
    
    return limits, prec_sum, time_sum, prec_n


