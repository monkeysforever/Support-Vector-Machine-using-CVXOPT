import numpy as np
from cvxopt import matrix, solvers

class SVC:
    """
    Implemenation of Support Vector binary classifier
    """
    def __init__(self, C = None, kernel = 'Linear', degree = 2, intercept = 1, sigma = 0.1, normalize = True):
        """
        Initialize classifier

        :param float C : Slack variable
        :param string kernel : Type of kernel
        :param integer degree : Degree for polynomial kernel
        :param float intercept : Intercept for polynomial kernel
        :param float sigma : Sigma for Gaussian kernel
        :param boolean normalize : To normalize data
        """        
        if kernel == 'Linear':
            self.kernel = self._linear_kernel
        elif kernel == 'Gaussian':
            self.kernel = self._gaussian_kernel
        elif kernel == 'Poly':
            self.kernel = self._poly_kernel
        else:
            raise Exception('Unsupported Kernel')
        
        self.C = C
        self.degree = degree
        self.intercept = intercept
        self.sigma = sigma
        self.normalize = normalize
    
    def fit(self, X, y):
        """
        Fit input data X to labels y

        :param numpy.array X : Input data
        :param numpy.array y : Labels, must be binary
        """
        m, n = X.shape
        X = np.copy(X)
        y = np.copy(y)
        y = y.reshape(-1, 1)

        assert(X.shape[0] == y.shape[0])
        self.labels = np.unique(y)
        print(self.labels)
        assert(len(self.labels) == 2)
        
        mask = y == self.labels[0]
        y[mask] = 1       
        y[~mask] = -1
        y = y * 1.
        
        if self.normalize:
            self.means = np.mean(X, axis = 0)
            self.stds = np.std(X, axis = 0)
            X = (X - self.means) / self.stds
        
        K = self.kernel(X, X)
        P = matrix(np.matmul(y,y.T) * K)
        q = matrix(np.ones((m, 1)) * -1)
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))

        if self.C is None:
            G = matrix(np.eye(m) * -1)
            h = matrix(np.zeros(m))
        else:            
            G = matrix(np.vstack((np.eye(m) * -1, np.eye(m))))        
            h = matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        
        solution = solvers.qp(P, q, G, h, A, b)
        assert(solution['status'] != 'unkown')

        alphas = np.array(solution['x'])        
        ind = (alphas > 1e-4).flatten()
        self.sv = X[ind]
        self.sv_y = y[ind]
        self.alphas = alphas[ind]
        b = self.sv_y - np.sum(self.kernel(self.sv, self.sv) * self.alphas * self.sv_y, axis=0)
        self.b = np.sum(b) / b.size
        
        if self.kernel == self._linear_kernel:            
            self.w = np.sum(self.alphas * self.sv_y * self.sv, axis=0)            
        else:
            self.w = None            
            

    def predict(self, X):
        """
        Predict labels for input data X

        :param numpy.array X : Input data
        :return np.array : Predictions
        """
        if self.normalize:
            X = (X - self.means) / self.stds
        if self.w is None:
            prod = np.sum(self.kernel(self.sv, X) * self.alphas * self.sv_y,
                      axis=0) + self.b
        else:
            prod = np.dot(self.w, X.T) + self.b
        y = np.sign(prod)
        y = y.reshape(-1, 1)
        mask = y == 1.
        y[mask] = self.labels[0]
        y[~mask] = self.labels[1]
        return y

    def _poly_kernel(self, x, z):
        """
        Polynomial Kernel

        :param numpy.array x
        :param numpy.array z
        :return numpy.array
        """
        return np.power(np.matmul(x, z.T) + self.intercept, self.degree)

    def _gaussian_kernel(self, x, z):
        """
        Gaussian Kernel

        :param numpy.array x
        :param numpy.array z
        :return numpy.array
        """
        n = x.shape[0]
        m = z.shape[0]
        xx = np.dot(np.sum(np.power(x, 2), 1).reshape(n, 1), np.ones((1, m)))
        zz = np.dot(np.sum(np.power(z, 2), 1).reshape(m, 1), np.ones((1, n)))     
        return np.exp(-(xx + zz.T - 2 * np.dot(x, z.T)) / (2 * self.sigma ** 2))

    def _linear_kernel(self, x, z):
        """
        Linear Kernel

        :param numpy.array x
        :param numpy.array z
        :return numpy.array
        """
        return np.matmul(x, z.T)