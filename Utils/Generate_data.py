from sklearn.datasets import make_moons, make_circles, make_blobs
import numpy as np

def generate_dataset(examples_count, dataset_type = 'moons', split_ratio = 90):
    """
    Generate training and test data for testing SVM

    :param integer examples_count : Number of training + testing examples
    :param string dataset_type : Type of dataset_type
    :param float split_ratio : ratio of training and testing examples
    :return numpy.array, numpy.array, numpy.array, numpy.array
    """   
    if dataset_type == 'blobs':
        X, y = make_blobs(n_samples=examples_count, centers=2, n_features=2)        
    elif dataset_type == 'moons':
        X, y = make_moons(n_samples=examples_count, noise=0.1)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=examples_count, noise=0.05)
        
    X = X.T
    y = np.reshape(y, (1, y.shape[0]))   
    split_index = X.shape[1] * split_ratio // 100
    indices = np.random.permutation(X.shape[1])
    training_idx, test_idx = indices[:split_index], indices[split_index:]    
    X_training, X_test, Y_training, Y_test = X[:, training_idx], X[:, test_idx], y[:, training_idx], y[:, test_idx]    
    return X_training, X_test, Y_training, Y_test