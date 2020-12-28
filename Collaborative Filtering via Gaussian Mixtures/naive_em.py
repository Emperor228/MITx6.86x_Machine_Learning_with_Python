"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    k, _ = mixture.mu.shape
    p_i_by_j = np.ndarray(shape = (n,k))
    for i in range(n):
        for j in range(k):
            pj = mixture.p[j]
            var = mixture.var[j]
            mu = mixture.mu[j]
            x = X[i] 
            pi = np.pi
            e = np.e
            p_i_by_j[i,j] = pj*((2*pi)**(-d/2))*(var**(-d/2))*(e**((-1/2)*np.dot(x-mu,x-mu)/var))
    sigma_p_i_by_j = p_i_by_j.sum(axis = 1)
    pji = (p_i_by_j.T / sigma_p_i_by_j.T).T
    loglike = np.log(sigma_p_i_by_j).sum()    
    return pji,loglike
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, k = post.shape
    mu = np.ndarray(shape = (k,d))
    var = np.ndarray(shape = (k,))
    p = np.ndarray(shape = (k,))
    count = post.sum(axis = 0)
    for j in range(k):
        nj = count[j]
        p[j] = nj/n
        
        for l in range(d):
            mu[j,l] = np.dot(post[:,j],X[:,l])/nj
        var[j] = 0
        
        for i in range(n):
            var[j] += (1/(nj*d))*post[i,j]*(np.linalg.norm(X[i]-mu[j])**2)
        
    mixture = GaussianMixture.__new__(GaussianMixture,mu = mu,var = var,p = p)

    return mixture
        
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    
    newlog = -np.inf
    while (True):
        oldlog = newlog
        post, newlog = estep(X,mixture)
        if (newlog - oldlog)<=abs(newlog)*0.000001:
            break
        mixture = mstep(X,post)
    return mixture, post, newlog 
    raise NotImplementedError
