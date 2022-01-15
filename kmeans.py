import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    p = generator.randint(0, n) #this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
	# according to some distribution, first generate a random number between 0 and
	# 1 using generator.rand(), then find the the smallest index n so that the 
	# cumulative probability from example 1 to example n is larger than r.
    #############################################################################   
    centers = [p]
    while len(centers) < n_cluster:
        distances = np.power(np.linalg.norm(x[:, np.newaxis] - x[centers], axis=2), 2)
        distances = np.min(distances, axis=1)[:, np.newaxis]
        distances_prob = distances / distances.sum(axis=0)
        cumsum = np.cumsum(distances_prob, axis=0)
        
        r = generator.rand()
        index = np.argmin((cumsum - r) < 0.0)
        centers.append(index)
        
    assert len(centers) == n_cluster

    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)



class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        centroids = x[self.centers]
        memberships = np.zeros((N, self.n_cluster))
        objective_value = None
        for t in range(self.max_iter):
            # E-Step Assign each point to the closest center
            memberships = np.linalg.norm(x[:, np.newaxis] - centroids, axis=2)
            memberships_indices = np.argmin(memberships, axis=1)
            memberships = np.zeros(memberships.shape)
            memberships[np.arange(len(memberships)), memberships_indices] = 1
            
            # M-Step: Update the centers
            centroids = (memberships.T @ x) / np.maximum(1, memberships.sum(axis=0)[:, np.newaxis])
            
            # Calculate objective
            distances = np.power(np.linalg.norm(x[:, np.newaxis] - centroids, axis=2), 2)
            cur_objective_value = (memberships * distances).sum()
            
            if objective_value and np.abs(objective_value - cur_objective_value) < self.e:
                break
            objective_value = cur_objective_value
        
        memberships = np.linalg.norm(x[:, np.newaxis] - centroids, axis=2)
        memberships_indices = np.argmin(memberships, axis=1)
        return centroids, memberships_indices, t

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        kmeans = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        centroids, memberships_indices, iterations = kmeans.fit(x, centroid_func)
        
        centroid_label_counter = [[] for x in range(self.n_cluster)]
        for centroid_index, label in zip(memberships_indices, y):
            centroid_label_counter[centroid_index].append(label)
        
        centroid_labels = np.zeros(self.n_cluster)
        for k in range(self.n_cluster):
            counts = np.bincount(centroid_label_counter[k])
            label = np.argmax(counts)
            centroid_labels[k] = label
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        memberships = np.linalg.norm(x[:, np.newaxis] - self.centroids, axis=2)
        memberships_indices = np.argmin(memberships, axis=1)
        predicted_labels = self.centroid_labels[memberships_indices]
        
        assert predicted_labels.shape == (N,)
        
        return predicted_labels




def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    
    # Flatten the pixels of the image to 1D
    image_1D = image.reshape(-1, 3)
    
    # Calculate the index of the nearest cluster for each pixel
    memberships = np.linalg.norm(image_1D[:, np.newaxis] - code_vectors, axis=2)
    memberships_indices = np.argmin(memberships, axis=1)
    
    # Retrieve the cluster position for each pixel
    compressed_image = code_vectors[memberships_indices]
    
    # Reconstruct the 2D image from the 1D compressed image
    result = compressed_image.reshape(image.shape)
    
    assert result.shape == image.shape
    
    return result
