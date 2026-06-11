import numpy as np
import matplotlib.pyplot as plt
import os
from neural_net import load_mnist  


def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def predict(X, params):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    
    if W1.shape[0] == 784:
        # Handles weights formatted as (784, hidden_dim)
        Z1 = np.dot(X, W1) + b1.ravel()
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2.ravel()
        
        return np.argmax(Z2, axis=1)
        
    else:
        # Handles weights formatted as (hidden_dim, 784)
        Z1 = np.dot(W1, X.T) + b1
        A1 = relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = softmax(Z2)
        return np.argmax(A2, axis=0)
    
    if W1.shape[0] == 784:
        Z1 = np.dot(X, W1) + b1.T
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2.T
        A2 = softmax(Z2.T) 
    else:
        Z1 = np.dot(W1, X.T) + b1
        A1 = relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = softmax(Z2)
    
    return np.argmax(A2, axis=0)

if __name__ == "__main__":
    # 1. Load the MNIST test dataset
    _, _, X_test, y_test = load_mnist()
    
    # 2. Check if the trained model parameters exist
    if not os.path.exists("data/params.npy"):
        print("Error: Model parameters not found. Please run train.py first to save the model!")
        exit()
        
    params = np.load("data/params.npy", allow_pickle=True).item()
    
    # Generate predictions for the test dataset
    predictions = predict(X_test, params)
    
    # 3. Randomly select and plot 4 test images with predictions
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()
    
    random_indices = np.random.choice(len(X_test), 4, replace=False)
    
    for i, idx in enumerate(random_indices):
        img = X_test[idx].reshape(28, 28)
        actual_label = y_test[idx]
        predicted_label = predictions[idx]
        
        axes[i].imshow(img, cmap='gray')
        
        # Display green title for correct predictions, red for incorrect ones
        color = 'green' if actual_label == predicted_label else 'red'
        axes[i].set_title(f"Actual: {actual_label} | Pred: {predicted_label}", color=color)
        axes[i].axis('off')
        
    plt.tight_layout()
    print("Displaying evaluation graph... Check the window to see the model predictions!")
    plt.show()