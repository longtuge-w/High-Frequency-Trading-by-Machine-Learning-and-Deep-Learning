import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def preprocess(data):
    # fill NaN
    data['opened_position_qty '].fillna(np.floor(data['transacted_qty']/2), inplace = True)
    data['closed_position_qty'].fillna(np.ceil(data['transacted_qty']/2), inplace = True)
    
    # add some features
    data['diff'] = data['ask1'] - data['bid1']
    data['diff_vol'] = data['bid1vol'] - data['ask1vol']
    data['pot_vol'] = data['bid1vol'] + data['ask1vol']
    data['frac_price'] = data['last_price'] / data['mid']
    data['ask_spread'] = data['ask1'] / data['ask5']
    data['ask_vol_spread'] = data[['ask1vol', 'ask2vol', 'ask3vol', 'ask4vol', 'ask5vol']].min(axis=1) / data[['ask1vol', 'ask2vol', 'ask3vol', 'ask4vol', 'ask5vol']].max(axis=1)
    data['bid_spread'] = data['bid1'] / data['bid5']
    data['bid_vol_spread'] = data[['bid1vol', 'bid2vol', 'bid3vol', 'bid4vol', 'bid5vol']].min(axis=1) / data[['bid1vol', 'bid2vol', 'bid3vol', 'bid4vol', 'bid5vol']].max(axis=1)
    
    # Convert the relevant portion of the DataFrame to a NumPy array for fast computation
    features_array = data.values
    # Compute the mean and std dev row-wise
    means = np.mean(features_array, axis=1, keepdims=True)
    std_devs = np.std(features_array, axis=1, keepdims=True)
    # Apply z-score standardization
    standardized_features = (features_array - means) / std_devs
    # Replace NaN values in the standardized array (resulting from division by zero if std_dev is 0)
    standardized_features = np.nan_to_num(standardized_features)
    # Put the standardized data back into the DataFrame
    data[:] = standardized_features

    return data