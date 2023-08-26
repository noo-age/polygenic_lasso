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
iters = 5 # iterates through losses_0.csv, etc.

data = np.loadtxt(directory + 'mydata_with_phenotypes.txt')

# Separate features and target
X = data[:, :-2]
y_measured = data[:, -1]
y_true = data[:,-2]

# Convert numpy arrays to PyTorch tensors
X = torch.from_numpy(X).float() # assuming data is in float format
y_measured = torch.from_numpy(y_measured).float()
y_true = torch.from_numpy(y_true).float()

def r_correlation(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")
    tensor1_mean = torch.mean(tensor1)
    tensor2_mean = torch.mean(tensor2)
    tensor1_centered = tensor1 - tensor1_mean
    tensor2_centered = tensor2 - tensor2_mean
    correlation = torch.sum(tensor1_centered * tensor2_centered) / (torch.sqrt(torch.sum(tensor1_centered ** 2)) * torch.sqrt(torch.sum(tensor2_centered ** 2)))
    return correlation.item()

def save_losses_to_csv(train_losses, val_losses, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.stat(filename).st_size == 0:
            writer.writerow(['train_loss', 'val_loss'])
        for i in range(len(train_losses)):
            writer.writerow([train_losses[i], val_losses[i]])

def save_correlation_to_csv(predicted, actual, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['predicted trait', 'actual trait'])
        for i in range(len(predicted)):
            writer.writerow([predicted[i].item(), actual[i].item()])

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
    plt.show(block=False)

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

def main():
    '''
    # Plot "true" phenotype as expected from genotype
    print("mean", torch.mean(y_true))
    print("sd", torch.std(y_true))
    plot_distribution(y_true)
    
    # Plot measured phenotype
    print("mean", torch.mean(y_measured))
    print("sd", torch.std(y_measured))
    plot_distribution(y_measured)
    '''
        
    for i in range(iters):
        # Plot losses and pred|actual pairs to csv
        plot_losses(directory + f'losses_{i}.csv')
        plot_correlation(directory + f'correlation_{i}.csv')

        

        
    
if __name__ == '__main__':
    main()