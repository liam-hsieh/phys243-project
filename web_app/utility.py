import numpy as np
from scipy.stats import triang
import matplotlib.pyplot as plt

def rv_generator(low_cost: float, high_cost: float, high_end: int, num: int)->np.ndarray:
    """
    Generate random variables using a triangular distribution.

    Args:
        low_cost (float): Lower limit of triangular distribution.
        high_cost (float): Upper limit of triangular distribution.
        high_end (int): Mode of triangular distribution expressed by a Scale Index between 1 and 10.
        num (int): Number of random numbers to be returned.

    Returns:
        np.ndarray: Array of random numbers generated from the triangular distribution.
    """
    # Calculate parameters for the triangular distribution
    a = low_cost
    b = high_cost
    c = a + (high_end - 1) * (b - a) / 9

    # Create a triangular distribution object
    triangular_dist = triang(c=(c - a) / (b - a), loc=a, scale=(b - a))

    # Generate random samples from the triangular distribution
    samples = triangular_dist.rvs(size=num)
    
    return samples


