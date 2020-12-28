import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    
    """ @author: YiWei """
    X_Yt = np.matmul(X,np.transpose(Y))
    kernel_matrix = (X_Yt+c)**p
    return kernel_matrix
    raise NotImplementedError
    """ @author: YiWei """



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    
    """ @author: YiWei """
    n = X.shape[0]
    m = Y.shape[0]
    kernel = np.zeros((n,m))
    for x_id in range(n):
        for y_id in range(m):
            kernel[x_id,y_id] = np.exp(-gamma*((np.linalg.norm(X[x_id] - Y[y_id]))**2))
    return kernel            
    raise NotImplementedError
    """ @author: YiWei """
