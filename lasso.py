import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

epochs = 1000
batch_size = 2
learning_rate = 0.01
l1_penalty = 0.00 # coefficient of penalty of weights

def train_test_split(X, y, test_size=0.2, random_state=None):
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
        self.linear = nn.Linear(n_features, 1)
        self.l1_penalty = l1_penalty

    def forward(self, x):
        return self.linear(x).squeeze()

    def lasso_loss(self, y_pred, y):
        return nn.MSELoss()(y_pred, y) + self.l1_penalty * torch.norm(self.linear.weight, 1)
    
    def generate(self, x): # takes normal list and returns model prediction
        return self(torch.tensor(x,dtype=torch.float)).item()

def train(model, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        batch_losses = []
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            y_pred = model(X_batch)
            loss = model.lasso_loss(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_losses.append(np.mean(batch_losses))

        # Validation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val)
            loss = model.lasso_loss(y_pred, y_val)
            val_losses.append(loss.item())

        print(f'Epoch {epoch+1}/{epochs} => Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}')

    return train_losses, val_losses

def save_losses_to_csv(train_losses, val_losses, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['train_loss', 'val_loss'])
        for i in range(len(train_losses)):
            writer.writerow([train_losses[i], val_losses[i]])

def plot_losses(filepath):
    data = pd.read_csv(filepath)
    
    # Plot train loss
    plt.plot(data['train_loss'], label='Train Loss')
    
    # Plot validation loss
    plt.plot(data['val_loss'], label='Validation Loss')
    
    # Labels for x and y axes
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Title of the graph
    plt.title('Train Loss vs Validation Loss')
    
    # Show legend
    plt.legend()
    
    # Display the plot
    plt.show()

def main():
    # Load data
    data = np.loadtxt('data.txt')

    # Separate features and target
    X = data[:, :-1]
    y = data[:, -1]

    # Convert numpy arrays to PyTorch tensors
    X = torch.from_numpy(X).float() # assuming data is in float format
    y = torch.from_numpy(y).float()

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model
    model = LassoRegression(X.shape[1], l1_penalty=l1_penalty)
    if os.path.isfile('model.pth') and input("load model: y/n") == 'y':
        model.load_state_dict(torch.load('model.pth'))

    # Train model
    train_losses, val_losses = train(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

    # Save model
    torch.save(model.state_dict(), 'model.pth')

    # Save losses to csv
    save_losses_to_csv(train_losses, val_losses, 'losses.csv')
    
    plot_losses('losses.csv')
    
    print(model.generate([1,2,300]))
    

if __name__ == '__main__':
    main()
