import numpy as np


def linear_kernel(**kwargs):
    """
        np.inner() is the same as (x1*x2).sum()
    """
    return np.inner(kwargs['x1'],kwargs['x2'])


def polynomial_kernel(const, pow, **kwargs):
    return (np.inner(kwargs['x1'],kwargs['x2']) + const) ** pow


def rbf_kernel(gamma,**kwargs):
    """
        Radial Basis Function kernel. The more the gamma
        is, the more influential difference in distance is.
        That is, the more difference there is, more clusters
        are going to appear in data: see this img for reference
        https://drek4537l1klr.cloudfront.net/serrano/v-9/Figures/09image121.png

        technically gamma = 1/sigma^2, where with lower sigma, the gamma increases
        and the rejection region also increases
        (https://towardsdatascience.com/radial-basis-function-rbf-kernel-the-go-to-kernel-acf0d22c798a)
    """
    # l2 norm
    dist = np.linalg.norm(kwargs['x1']-kwargs['x2']) ** 2
    return np.exp(-gamma*dist)