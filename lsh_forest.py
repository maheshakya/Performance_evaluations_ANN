import numpy as np
from sklearn.metrics import euclidean_distances

#Re-implementation of bisect functions of bisect module to suit the application
def bisect_left(a, x):
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo
            
def bisect_right(a, x):
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x < a[mid] and not a[mid][:len(x)]==x:
            hi = mid
        else:
            lo = mid + 1
    return lo

#function which accepts an sorted array of bit strings, a query string
#This returns an array containing all indices which share the first h bits of the query
def simpleFunctionBisectReImplemented(sorted_array, item, h):
    left_index = bisect_left(sorted_array, item[:h])
    right_index = bisect_right(sorted_array, item[:h])
    return np.arange(left_index, right_index)

def find_longest_prefix_match(bit_string_list, query):
    hi = len(query)
    lo = 0
    
    if len(simpleFunctionBisectReImplemented(bit_string_list, query, hi)) > 0:
        return hi
    
    while lo < hi:
        mid = (lo+hi)//2        
        k = len(simpleFunctionBisectReImplemented(bit_string_list, query, mid))
        if k > 0:
            lo = mid + 1
            res = mid
        else:
            hi = mid            
        
    return res

class LSH_forest(object):
    """
    LSH forest implementation using numpy sorted arrays and 
    binary search. This is an initial effor for a hack. 
    NOT the final version.
    
    attributes
    ----------
    max_label_lengh: maximum length of hash 
    number_of_trees: number of trees build in indexing
    """
    def __init__(self, max_label_length = 32, number_of_trees = 5):
        self.max_label_length = max_label_length
        self.number_of_trees = number_of_trees
        self.min_label_length = 20
        self.random_state = np.random.RandomState(seed=1)
        
    def _get_random_hyperplanes(self, hash_size = None, dim = None):
        """ 
        Generates hyperplanes from standard normal distribution  and return 
        it as a 2D numpy array. This is g(p,x) for a particular tree.
        """
        if hash_size == None or dim == None:
            raise ValueError("hash_size or dim(number of dimensions) cannot be None.")        
        
        return self.random_state.randn(hash_size, dim) 
    
    def _hash(self, input_point = None, hash_function = None):
        """
        Does hash on the data point with the provided hash_function: g(p,x).
        """
        if input_point == None or hash_function == None:
            raise ValueError("input_point or hash_function cannot be None.")
            
        projections = np.dot(hash_function, input_point) 
            
        return "".join(['1' if i > 0 else '0' for i in projections])
    
    def _create_tree(self, input_array = None, hash_function = None):
        """
        Builds a single tree (in this case creates a sorted array of 
        binary hashes).
        """
        if input_array == None or hash_function == None:
            raise ValueError("input_array or hash_funciton cannot be None.")
            
        number_of_points = input_array.shape[0]
        binary_hashes = []
        for i in range(number_of_points):
            binary_hashes.append(self._hash(input_array[i], hash_function))
        
        binary_hashes = np.array(binary_hashes)
        o_i = np.argsort(binary_hashes)
        return o_i, np.sort(binary_hashes)
    
    def _compute_distances(self, query, candidates):
        distances = euclidean_distances(query, self.input_array[candidates])
        return np.argsort(distances), distances
        
        
    def build_index(self, input_array = None):
        """
        Builds index.
        """ 
        if input_array == None:
            raise ValueError("input_array cannot be None")
            
        self.input_array = np.array(input_array)
        number_of_points = input_array.shape[0]
        dim = input_array.shape[1]
        
        #Creates a g(p,x) for each tree
        self.hash_functions = []
        self.trees = []
        self.original_indices = []
        for i in range(self.number_of_trees):
            """"
            hash_size = self.random_state.randint(self.min_label_length, 
                                                  self.max_label_length+1)
            """
            hash_size = self.max_label_length
            hash_function = self._get_random_hyperplanes(hash_size = hash_size, dim = dim)
            o_i, bin_hashes = self._create_tree(input_array, hash_function)
            self.original_indices.append(o_i)
            self.trees.append(bin_hashes)
            self.hash_functions.append(hash_function)
        
        self.hash_functions = np.array(self.hash_functions)
        self.trees = np.array(self.trees)
        self.original_indices = np.array(self.original_indices)
        
    def query(self, query = None, c = 1, m = 10):
        """
        returns the number of neighbors for a given query.
        """
        if query == None:
            raise ValueError("query cannot be None.")
        query = np.array(query)
        
        #descend phase
        max_depth = 0
        for i in range(len(self.trees)):
            bin_query = self._hash(query, self.hash_functions[i])
            k = find_longest_prefix_match(self.trees[i], bin_query)
            if k > max_depth:
                max_depth = k
                
        #Asynchronous ascend phase
        candidates = []
        number_of_candidates = c*len(self.trees)
        while max_depth > 0 and (len(candidates) < number_of_candidates or len(set(candidates)) < m):
            for i in range(len(self.trees)):
                bin_query = self._hash(query, self.hash_functions[i])
                candidates.extend(self.original_indices[i,simpleFunctionBisectReImplemented(self.trees[i], 
                                                                                            bin_query, max_depth)].tolist())
                #candidates = list(OrderedSet(candidates)) #this keeps the order inserted into the list 
            max_depth = max_depth - 1
            #print max_depth, len(candidates) ,len(set(candidates))
        candidates = np.array(list(set(candidates)))
        ranks, distances = self._compute_distances(query, candidates)
        #print ranks[0,:m]
        print candidates.shape
        return candidates[ranks[0,:m]]

    def query_num_candidates(self, query = None, c = 1, m = 10):
        """
        returns the number of neighbors for a given query.
        """
        if query == None:
            raise ValueError("query cannot be None.")
        query = np.array(query)
        
        #descend phase
        max_depth = 0
        for i in range(len(self.trees)):
            bin_query = self._hash(query, self.hash_functions[i])
            k = find_longest_prefix_match(self.trees[i], bin_query)
            if k > max_depth:
                max_depth = k
                
        #Asynchronous ascend phase
        candidates = []
        number_of_candidates = c*len(self.trees)
        while max_depth > 0 and (len(candidates) < number_of_candidates or len(set(candidates)) < m):
            for i in range(len(self.trees)):
                bin_query = self._hash(query, self.hash_functions[i])
                candidates.extend(self.original_indices[i,simpleFunctionBisectReImplemented(self.trees[i], 
                                                                                            bin_query, max_depth)].tolist())
                #candidates = list(OrderedSet(candidates)) #this keeps the order inserted into the list 
            max_depth = max_depth - 1
            #print max_depth, len(candidates) ,len(set(candidates))
        candidates = np.array(list(set(candidates)))
        ranks, distances = self._compute_distances(query, candidates)
        #print ranks[0,:m]        
        return candidates[ranks[0,:m]], candidates.shape[0]
                
                

            
        
    

                