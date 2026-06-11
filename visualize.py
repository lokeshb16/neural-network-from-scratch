import numpy as np
import matplotlib.pyplot as plt
import os
from neural_net import load_mnist  # Humne jo pehle data loading ka function banaya tha

# Forward pass ke liye activation function
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    # Z agar 2D array hai toh columns ka sum nikalne ke liye axis=0 use hota hai
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# Model ke saved parameters load karne ka function
def predict(X, params):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    
    # Agar weights (784, 128) format mein hain
    if W1.shape[0] == 784:
        # b1.ravel() karne se woh simple 1D array ban jayega, 
        # jise NumPy har row (10000) mein aaram se add kar dega.
        Z1 = np.dot(X, W1) + b1.ravel()
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2.ravel()
        # Predictions nikalne ke liye axis=1 (row-wise max) use karenge
        return np.argmax(Z2, axis=1)
        
    # Agar weights (128, 784) format mein hain
    else:
        Z1 = np.dot(W1, X.T) + b1
        A1 = relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = softmax(Z2)
        return np.argmax(A2, axis=0)
    
    # FORWARD PASS FIX: 
    # Agar W1 ka shape (784, 128) hai, toh hum X (10000, 784) ko W1 se multiply karenge
    # Taaki output (10000, 128) bane, fir b1 add ho ske.
    if W1.shape[0] == 784:
        Z1 = np.dot(X, W1) + b1.T
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2.T
        A2 = softmax(Z2.T) # Softmax columns pe apply karne ke liye transpose kiya
    else:
        # Agar W1 ka shape already (128, 784) hai
        Z1 = np.dot(W1, X.T) + b1
        A1 = relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = softmax(Z2)
    
    return np.argmax(A2, axis=0)

if __name__ == "__main__":
    # 1. Data aur Saved Weights Load karo
    _, _, X_test, y_test = load_mnist()
    
    if not os.path.exists("data/params.npy"):
        print("Error: Pehle neural_net.py chalakari model save karo!")
        exit()
        
    params = np.load("data/params.npy", allow_pickle=True).item()
    
    # 2. Predictions nikalo
    predictions = predict(X_test, params)
    
    # 3. Randomly 4 images pick karke plot karo
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()
    
    # Koi bhi 4 random test images uthate hain
    random_indices = np.random.choice(len(X_test), 4, replace=False)
    
    for i, idx in enumerate(random_indices):
        img = X_test[idx].reshape(28, 28)
        actual_label = y_test[idx]
        predicted_label = predictions[idx]
        
        axes[i].imshow(img, cmap='gray')
        # Agar sahi predict kiya toh Green text, galat kiya toh Red text
        color = 'green' if actual_label == predicted_label else 'red'
        axes[i].set_title(f"Actual: {actual_label} | Pred: {predicted_label}", color=color)
        axes[i].axis('off')
        
    plt.tight_layout()
    print("Graph display ho raha hai... Dekho aapke AI ne sahi pehchana ya nahi!")
    plt.show()