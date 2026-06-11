import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────
def load_mnist():
    from urllib.request import urlretrieve
    import os, gzip

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }
    os.makedirs("data", exist_ok=True)
    for key, fname in files.items():
        fpath = os.path.join("data", fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname}...")
            urlretrieve(base_url + fname, fpath)

    def read_images(path):
        with gzip.open(path, 'rb') as f:
            f.read(16)
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(-1, 784) / 255.0

    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            f.read(8)
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

    X_train = read_images("data/train-images-idx3-ubyte.gz")
    y_train = read_labels("data/train-labels-idx1-ubyte.gz")
    X_test  = read_images("data/t10k-images-idx3-ubyte.gz")
    y_test  = read_labels("data/t10k-labels-idx1-ubyte.gz")
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────
# 2. ACTIVATION FUNCTIONS
# ─────────────────────────────────────────
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)


# ─────────────────────────────────────────
# 3. WEIGHT INITIALIZATION
# ─────────────────────────────────────────
def initialize_weights():
    np.random.seed(42)
    return {
        "W1": np.random.randn(784, 128) * 0.01,
        "b1": np.zeros((1, 128)),
        "W2": np.random.randn(128, 10) * 0.01,
        "b2": np.zeros((1, 10))
    }


# ─────────────────────────────────────────
# 4. FORWARD PASS
# ─────────────────────────────────────────
def forward_pass(X, params):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "X": X}
    return A2, cache


# ─────────────────────────────────────────
# 5. LOSS + ACCURACY
# ─────────────────────────────────────────
def compute_loss(A2, y):
    m = y.shape[0]
    correct_probs = A2[np.arange(m), y]
    return -np.mean(np.log(correct_probs + 1e-8))

def compute_accuracy(A2, y):
    predictions = np.argmax(A2, axis=1)
    return np.mean(predictions == y)


# ─────────────────────────────────────────
# 6. BACKWARD PASS ← NEW ✨
# ─────────────────────────────────────────
def backward_pass(cache, params, y):
    m = y.shape[0]
    A1, A2 = cache["A1"], cache["A2"]
    Z1      = cache["Z1"]
    X       = cache["X"]
    W2      = params["W2"]

    # Output layer ka error
    dZ2 = A2.copy()
    dZ2[np.arange(m), y] -= 1       # correct class se 1 minus karo
    dZ2 /= m

    # Layer 2 gradients
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer ka error (backward propagate)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)

    # Layer 1 gradients
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


# ─────────────────────────────────────────
# 7. WEIGHTS UPDATE ← NEW ✨
# ─────────────────────────────────────────
def update_weights(params, grads, lr=0.1):
    params["W1"] -= lr * grads["dW1"]
    params["b1"] -= lr * grads["db1"]
    params["W2"] -= lr * grads["dW2"]
    params["b2"] -= lr * grads["db2"]
    return params


# ─────────────────────────────────────────
# 8. TRAINING LOOP ← NEW ✨
# ─────────────────────────────────────────
def train(X_train, y_train, X_test, y_test,
          epochs=20, batch_size=256, lr=0.1):

    params = initialize_weights()
    history = {"loss": [], "acc": [], "val_acc": []}

    print(f"Training started! Epochs: {epochs}, Batch: {batch_size}, LR: {lr}\n")
    print(f"{'Epoch':<8} {'Loss':<10} {'Train Acc':<12} {'Test Acc':<10}")
    print("-" * 42)

    for epoch in range(1, epochs + 1):
        # Shuffle karo har epoch mein
        indices = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Mini-batch training
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            A2, cache = forward_pass(X_batch, params)
            grads = backward_pass(cache, params, y_batch)
            params = update_weights(params, grads, lr)

        # Epoch end mein metrics calculate karo
        A2_train, _ = forward_pass(X_train, params)
        A2_test,  _ = forward_pass(X_test,  params)

        loss    = compute_loss(A2_train, y_train)
        acc     = compute_accuracy(A2_train, y_train)
        val_acc = compute_accuracy(A2_test,  y_test)

        history["loss"].append(loss)
        history["acc"].append(acc)
        history["val_acc"].append(val_acc)

        print(f"{epoch:<8} {loss:<10.4f} {acc*100:<12.2f} {val_acc*100:<10.2f}")

    print("\n✅ Training Complete!")
    print(f"   Final Train Accuracy : {history['acc'][-1]*100:.2f}%")
    print(f"   Final Test  Accuracy : {history['val_acc'][-1]*100:.2f}%")

    return params, history


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    params, history = train(X_train, y_train, X_test, y_test)

    # Results save karo (Step 4 mein use honge)
    np.save("data/params.npy", params)
    np.save("data/history.npy", history)
    print("\n💾 Model saved to data/params.npy")