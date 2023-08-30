import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import csv
import os

def r_correlation(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")
    tensor1_mean = torch.mean(tensor1)
    tensor2_mean = torch.mean(tensor2)
    tensor1_centered = tensor1 - tensor1_mean
    tensor2_centered = tensor2 - tensor2_mean
    correlation = torch.sum(tensor1_centered * tensor2_centered) / (torch.sqrt(torch.sum(tensor1_centered ** 2)) * torch.sqrt(torch.sum(tensor2_centered ** 2)))
    return correlation.item()

def r_squared_from_file(filepath, variable1, variable2):
    # Read the csv file
    df = pd.read_csv(filepath)

    # Convert the variables to tensors
    tensor1 = torch.tensor(df[variable1].values)
    tensor2 = torch.tensor(df[variable2].values)

    return r_correlation(tensor1, tensor2) ** 2

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

def plot_distribution(filepath, variable):
    # Read the csv file
    df = pd.read_csv(filepath)

    # Plot the distribution of the variable
    plt.figure(figsize=(10, 6))
    plt.hist(df[variable], bins=30, edgecolor='black')
    plt.title('Distribution of ' + variable)
    plt.xlabel(variable)
    plt.ylabel('Frequency')
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
    plot_distribution("simulated_phenotypes.csv", "genetic_component")
    plot_distribution("simulated_phenotypes.csv", "environmental_noise")
    plot_distribution("simulated_phenotypes.csv", "true_phenotype")
    plot_distribution("simulated_phenotypes.csv", "measurement_noise")
    plot_distribution("simulated_phenotypes.csv", "observed_phenotype")
    plot_distribution("simulated_phenotypes.csv", "maf_values")
    print(r_squared_from_file("simulated_phenotypes.csv","genetic_component","observed_phenotype"))
    
    
if __name__ == '__main__':
    main()