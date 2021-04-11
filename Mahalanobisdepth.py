import numpy as np


def mahalanobis(x, y, cov=None):

    """
   
    This function to Find the depth of Tree 
    
    The Mahalanobis depth (MD) function is proposed by Liu and Singh18 and it is based
on the well-known Mahalanobis distance de¯ned by Mahalanobis.19 MD of a point x
with respect to a dataset X is de¯ned as
    
    """
    x_mean = np.mean(x)
    Covariance = np.cov(np.transpose(y))
    inv_covmat = np.linalg.inv(Covariance)
    x_minus_mn = x - x_mean
    D_square = np.dot(np.dot(x_minus_mn, inv_covmat), np.transpose(x_minus_mn))
    return  " Mahalanobis depth (MD) {}".format(D_square)