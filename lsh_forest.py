import numpy as np
from sklearn.metrics import euclidean_distances
#from binary_searches import simpleFunctionBisectReImplemented, find_longest_prefix_match


#Re-implementation of bisect functions of bisect module to suit the application
def bisect_left(a, x):
    """
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo
    """
    return np.searchsorted(a, x)
            
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

def binary_search_string_equality(str1, str2):
    hi = len(str1)
    lo = 0
    while lo < hi:
        mid = (hi+lo)//2        
        if str1[:mid] == str2[:mid]:
            lo = mid + 1
        else:
            hi = mid
    return lo-1

def get_longest_prefix_length(bit_strings_array, query):
    pos = np.searchsorted(bit_strings_array, query)
    return np.max([binary_search_string_equality(bit_strings_array[pos-1],query), 
                   binary_search_string_equality(bit_strings_array[pos], query)])

def get_longest_prefix_length_with_position(bit_strings_array, query):
    pos = np.searchsorted(bit_strings_array, query)
    return pos, np.max([binary_search_string_equality(bit_strings_array[pos-1],query), 
                        binary_search_string_equality(bit_strings_array[pos], query)])

def simple_euclidean_distance(query, candidates):
    distances = np.zeros(candidates.shape[0])    
    for i in range(candidates.shape[0]):        
        distances[i] = np.linalg.norm(candidates[i]-query)        
    return distances

def get_positional_cands(length, pos, num):
    cands = []
    if num == 0:
        cands.append(pos)
        return cands
    if pos+num < length:
        cands.append(pos+num)
    if pos-num >= 0:
        cands.append(pos-num)
        
    return cands


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
        #distances = euclidean_distances(query, self.input_array[candidates])
        distances = simple_euclidean_distance(query, self.input_array[candidates])
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
            hash_function = self._get_random_hyperplanes(hash_size = self.max_label_length, dim = dim)
            o_i, bin_hashes = self._create_tree(input_array, hash_function)
            self.original_indices.append(o_i)
            self.trees.append(bin_hashes)
            self.hash_functions.append(hash_function)
        
        self.hash_functions = np.array(self.hash_functions)
        self.trees = np.array(self.trees)
        self.original_indices = np.array(self.original_indices)
        
    def query(self, query = None, c = 1, m = 10, lower_bound = 4):
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
            k = get_longest_prefix_length(self.trees[i], bin_query)
            if k > max_depth:
                max_depth = k
        
        bin_queries = []
        for i in range(len(self.trees)):
            bin_queries.append(self._hash(query, self.hash_functions[i]))

        #Synchronous ascend phase
        candidates = []
        number_of_candidates = c*len(self.trees)
        while max_depth > lower_bound and (len(candidates) < number_of_candidates or len(set(candidates)) < m):
            for i in range(len(self.trees)):                
                candidates.extend(self.original_indices[i,simpleFunctionBisectReImplemented(self.trees[i], 
                                                                                            bin_queries[i], max_depth)].tolist())
                #candidates = list(OrderedSet(candidates)) #this keeps the order inserted into the list 
            max_depth = max_depth - 1
            #print max_depth, len(candidates) ,len(set(candidates))
        candidates = np.unique(candidates)
        ranks, distances = self._compute_distances(query, candidates)
        #print ranks[0,:m]
        print candidates.shape
        return candidates[ranks[:m]]
    
    
    def query_num_candidates(self, query = None, c = 1, m = 10, lower_bound = 4):
        """
        returns the nearest neighbors for a given query the number of required 
        candidates.
        """
        if query == None:
            raise ValueError("query cannot be None.")
        query = np.array(query)
        
        #descend phase
        max_depth = 0
        for i in range(len(self.trees)):
            bin_query = self._hash(query, self.hash_functions[i])
            k = get_longest_prefix_length(self.trees[i], bin_query)
            if k > max_depth:
                max_depth = k
                
        bin_queries = []
        for i in range(len(self.trees)):
            bin_queries.append(self._hash(query, self.hash_functions[i]))
                
        #Synchronous ascend phase
        candidates = []
        number_of_candidates = c*len(self.trees)
        while max_depth > lower_bound and (len(candidates) < number_of_candidates or len(set(candidates)) < m):
            for i in range(len(self.trees)):
                candidates.extend(self.original_indices[i,simpleFunctionBisectReImplemented(self.trees[i], 
                                                                                            bin_queries[i], max_depth)].tolist())
                #candidates = list(OrderedSet(candidates)) #this keeps the order inserted into the list 
            max_depth = max_depth - 1
            #print max_depth, len(candidates) ,len(set(candidates))
        candidates = np.unique(candidates)
        ranks, distances = self._compute_distances(query, candidates)
        #print ranks[0,:m]        
        return candidates[ranks[:m]], candidates.shape[0]


    def query_candidates(self, query = None, c = 1, m = 10, lower_bound = 4):
        """
        returns the nearest neighbors for a given query the number of required 
        candidates.
        """
        if query == None:
            raise ValueError("query cannot be None.")
        query = np.array(query)
        
        #descend phase
        max_depth = 0
        for i in range(len(self.trees)):
            bin_query = self._hash(query, self.hash_functions[i])
            k = get_longest_prefix_length(self.trees[i], bin_query)
            if k > max_depth:
                max_depth = k

        bin_queries = []
        for i in range(len(self.trees)):
            bin_queries.append(self._hash(query, self.hash_functions[i]))  

        #Synchronous ascend phase
        candidates = []
        number_of_candidates = c*len(self.trees)
        while max_depth > lower_bound and (len(candidates) < number_of_candidates or len(set(candidates)) < m):
            for i in range(len(self.trees)):            
                candidates.extend(self.original_indices[i,simpleFunctionBisectReImplemented(self.trees[i], 
                                                                                            bin_queries[i], max_depth)].tolist())
                #candidates = list(OrderedSet(candidates)) #this keeps the order inserted into the list 
            max_depth = max_depth - 1
            #print max_depth, len(candidates) ,len(set(candidates))
        candidates = np.unique(candidates)
        ranks, distances = self._compute_distances(query, candidates)
        #print ranks[0,:m]        
        return candidates[ranks[:m]], candidates


    def get_candidates_for_hash_length(self, query, hash_length):
        candidates = []        
        for i in range(len(self.trees)):
            bin_query = self._hash(query, self.hash_functions[i])
            candidates.extend(self.original_indices[i,simpleFunctionBisectReImplemented(self.trees[i], 
                                                                                            bin_query, hash_length)].tolist())
            
        return np.unique(candidates)