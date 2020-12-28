"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    k, _ = mixture.mu.shape
    lp_i_by_j = np.ndarray(shape = (n,k))
    CX = (X!=0)
    sigma_lp_i_by_j = np.ndarray(shape = (n,))
    
    for i in range(n):
        for j in range(k):
            lpj = np.log(mixture.p[j]+ 1e-16)
            var = mixture.var[j]
            lvar = np.log(var)
            
            x = X[i] 
            cx = CX[i]
            lcxl = cx.sum()
            mu = mixture.mu[j]*cx
            l2pi = np.log(np.pi*2)
            lp_i_by_j[i,j] = lpj-(l2pi*(lcxl/2))-(lvar*(lcxl/2))-(((1/2)*np.dot(x-mu,x-mu)/var))
        sigma_lp_i_by_j[i] = logsumexp(lp_i_by_j[i,:])
        
    lpji = (lp_i_by_j.T - sigma_lp_i_by_j).T
    loglike = sigma_lp_i_by_j.sum()    
    return np.exp(lpji),loglike
    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, k = post.shape
    mu = mixture.mu
    var = np.ndarray(shape = (k,))
    p = np.ndarray(shape = (k,))
    delta = (X != 0)
    for j in range(k):
        sse = 0
        post_j = post[:,j]
        deno_of_sigsqr = 0
       
        for l in range(d):
            delta_l = delta[:,l]
            masked_post_jl = post_j*delta_l
            count_lj = masked_post_jl.sum()
            if count_lj>=1:
                weighted_x_lj = np.dot(masked_post_jl,X[:,l])
                mu[j,l] = weighted_x_lj/count_lj
            deno_of_sigsqr += count_lj
            

            se = (((X[:,l]-mu[j,l])**2)*delta[:,l])
            sse += np.dot(se,post[:,j])
        var[j] = sse/deno_of_sigsqr
        if var[j] < min_variance:
            var[j] = min_variance
        p[j] = post_j.sum()/n
                
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
        mixture = mstep(X,post,mixture)
    return mixture, post, newlog 
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    new_x = X
    post, _ = estep(X,mixture)
    mask = np.where(new_x == 0)
    mu = mixture.mu
    new_x[mask] = (post @ mu)[mask]
    return new_x
    raise NotImplementedError
