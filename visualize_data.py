import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import csv
import os

directory = 'Models/lasso_firstsim/'

data = np.loadtxt(directory + 'mydata_with_phenotypes.txt')

def plot_correlation(filepath):
    # Read csv file
    data = pd.read_csv(filepath)
    
    # Plot the data
    plt.plot(data['predicted trait'], data['actual trait'], 'o')
    
    # Setting labels and title
    plt.xlabel('Predicted Trait')
    plt.ylabel('Actual Trait')
    plt.title('Predicted vs Actual Trait')

    # Show the plot
    plt.show()

def plot_distribution(scores):
    plt.figure(figsize=(10,6))
    sns.distplot(scores, hist = False, kde = True, 
                 kde_kws = {'shade': True, 'linewidth': 3})
    plt.title('Distribution of scores')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.show()
    
def plot_losses(filepath):
    data = pd.read_csv(filepath)
    
    # Get the labels from the first row of the data
    labels = data.columns
    
    for label in labels:
        # Skip the label if it is not numeric
        if not pd.to_numeric(data[label], errors='coerce').notnull().all():
            continue
        
        # Plot the data for the label
        plt.plot(data[label], label=label)
    
    # Labels for x and y axes
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Title of the graph
    plt.title('Losses')
    
    # Show legend
    plt.legend()
    
    # Display the plot
    plt.show()

# Separate features and target
X = data[:, :-2]
y_measured = data[:, -1]
y_true = data[:,-2]

# Convert numpy arrays to PyTorch tensors
X = torch.from_numpy(X).float() # assuming data is in float format
y_measured = torch.from_numpy(y_measured).float()
y_true = torch.from_numpy(y_true).float()

# Plot losses and pred|actual pairs to csv
plot_losses(directory + 'losses.csv')
plot_correlation(directory + 'correlation.csv')

# Plot measured phenotype
print("mean", torch.mean(y_measured))
print("sd", torch.std(y_measured))
plot_distribution(y_measured)

# Plot "true" phenotype as expected from genotype
print("mean", torch.mean(y_true))
print("sd", torch.std(y_true))
plot_distribution(y_true)