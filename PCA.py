import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def standardize_data(X_train, X_test):
    # We are standardizing the data (centering the data around mean 0 and standard deviation 1) 
    # This is to make sure certain features don't dominate (and thus skew) the results due to their scale
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std

def covariance_matrix(X):
    # Calculate the covariance matrix
    # Covariance matrix helps us understand how the features vary with respect to each other
    cov_mat = np.cov(X.T)
    return cov_mat


def main():
    # Load dataset
    dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    
    # get the locations
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=0)
    
    # Standardize the data
    X_train_std, X_test_std = standardize_data(X_train, X_test)

    data = np.array(X_train_std, y_train)

    #Get the covariance matrix
    cov_mat = covariance_matrix(data)
    
main()