import numpy as np
import pandas as pd
from scipy.stats import triang
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import os
import multiprocessing

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



def boostraping(X, y, features_to_predict):
    indices = np.random.choice(len(X), len(X), replace=True)
    X_sample = X.iloc[indices]
    y_sample = y.iloc[indices]
    
    clf = Ridge(alpha=1.0)
    clf.fit(X_sample, y_sample)
    return clf.predict(features_to_predict) 

def pseudo_data(data, high_end: int):
    result_rows = []

    # Iterate through each row in the original DataFrame
    for idx, row in data.iterrows():
        samples = rv_generator(row['low cost'], row['high cost'], high_end, 10)
        for sample in samples:
            result_rows.append({
                'bedrooms': row['bedrooms'],
                'bathrooms': row['bathrooms'],
                'kitchen': row['kitchen'],
                'living room': row['living room'],
                'detached': row['detached'],
                'modified sqft': row['modified sqft'],
                'additional sqft': row['additional sqft'],
                '2nd story': row['2nd story'],
                'cost': sample
            })

    # Create a DataFrame from the list of dictionaries
    return pd.DataFrame(result_rows)

def cost_distribution_estimate(user_input,high_end, num_samples = 1000):
    
    def generate_predicted_values(seed):
        np.random.seed(seed)
        return boostraping(X, y,[user_input])
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, 'price_data.csv')
    raw_price_data = pd.read_csv(csv_file_path)
    
    # generate pseudo data with given high_end
    df = pseudo_data(raw_price_data, high_end=high_end)
    X = df.drop(['cost'], axis=1)
    y = df['cost']

    # Initialize an array to store the predicted values
    predicted_values = np.zeros((num_samples))

    # Use multiprocessing to generate predicted values in parallel
    with multiprocessing.Pool() as pool:
        results = pool.map(generate_predicted_values, range(num_samples))
        predicted_values = np.array(results)

    mean_predicted_value = np.mean(predicted_values)
    return mean_predicted_value, predicted_values


def cost_distribution_estimate_non_mp_version(user_input,high_end, num_samples = 1000):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, 'price_data.csv')
    raw_price_data = pd.read_csv(csv_file_path)
    
    # generate pseudo data with given high_end
    df = pseudo_data(raw_price_data, high_end=high_end)
    X = df.drop(['cost'], axis=1)
    y = df['cost']

    predicted_values = np.array([boostraping(X,y,[user_input]) for _ in range(num_samples)])
    mean_predicted_value = np.mean(predicted_values)
    
    return mean_predicted_value, predicted_values