import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pysnptools.snpreader import Bed
import matplotlib.pyplot as plt
import seaborn as sns
import visualize_data as vd
import math
import csv
import os

epochs = 50
batch_size = 32
learning_rate = 0.0003
l1_penalties = [0.001, 0.01, 0.1] # coefficient of penalty of weights
val_size = 0.2
k = 3 # k-fold cross validation

directory = 'Models/lasso_firstsim/'

# Load data
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

def train_test_split(X, y, test_size=val_size, random_state=None):
    # Set the seed for reproducibility
    if random_state is not None:
        torch.manual_seed(random_state)

    # Shuffle the indices
    indices = torch.randperm(X.size(0))

    # Calculate the number of test samples
    test_count = int(test_size * X.size(0))

    # Split the indices for train and test
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    # Create train and test sets
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def get_batches(X, y, batch_size):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        batch_idx = indices[start:end]
        
        yield X[batch_idx, :], y[batch_idx]

class LassoRegression(nn.Module):
    def __init__(self, n_features, l1_penalty):
        super(LassoRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1,bias=False)
        self.l1_penalty = l1_penalty

    def forward(self, x):
        return self.linear(x).squeeze()

    def lasso_loss(self, y_pred, y):
        return nn.MSELoss()(y_pred, y) + self.l1_penalty * torch.norm(self.linear.weight, 1)
    
    def generate(self, x): # takes normal list and returns model prediction
        return self(torch.tensor(x,dtype=torch.float)).item()
    
    def print_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)     

def train(model, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        batch_trainlosses = []
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            y_pred = model(X_batch)
            loss = model.lasso_loss(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_trainlosses.append(loss.item())
        train_losses.append(np.mean(batch_trainlosses))

        # Validation
        model.eval()
        batch_vallosses = []
        with torch.no_grad():
            for X_valbatch, y_valbatch in get_batches(X_val, y_val, batch_size):
                y_pred = model(X_valbatch)
                loss = model.lasso_loss(y_pred, y_valbatch)
                batch_vallosses.append(loss.item())
            val_losses.append(np.mean(batch_vallosses))

        print(f'Epoch {epoch+1}/{epochs} => Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}')

    return train_losses, val_losses

def main():
    for i in range(k):
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y_measured, test_size=val_size, random_state=41)

        # Load model
        model = LassoRegression(X.shape[1], l1_penalty=l1_penalties[i])
        model_file = directory + f'model_{i}.pth'
        if os.path.isfile(model_file) and input("load model: y/n") == 'y':
            model.load_state_dict(torch.load(model_file))

        # Train model
        train_losses, val_losses = train(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

        # Save model
        torch.save(model.state_dict(), model_file)

        # Save losses and pred|actual pairs to csv
        vd.save_losses_to_csv(train_losses, val_losses, directory + f'losses_{i}.csv')
        vd.save_correlation_to_csv(model(X), y_measured, directory + f'correlation_{i}.csv')

        # Print model weights
        model.print_weights()

        
        # Plot losses and pred|actual pairs to csv
        vd.plot_losses(directory + f'losses_{i}.csv')
        vd.plot_correlation(directory + f'correlation_{i}.csv')

        # Plot measured phenotype
        vd.plot_distribution(y_measured, file_name=directory + f'y_measured_{i}.png')

        # Plot "true" phenotype as expected from genotype
        vd.plot_distribution(y_true, file_name=directory + f'y_true_{i}.png')

        phen_gen = r_correlation(y_measured,y_true)
        predgen_phen = r_correlation(model(X),y_measured)
        predgen_gen = r_correlation(model(X),y_true)
        print(f"For iteration {i}:")
        print("r, r^2 between genotype and phenotype:", phen_gen, phen_gen ** 2)
        print("r, r^2 between predicted phenotype and phenotype:", predgen_phen, predgen_phen ** 2)
        print("r, r^2 between predicted phenotype and genotype:", predgen_gen, predgen_gen ** 2)
        
    
if __name__ == '__main__':
    main()
