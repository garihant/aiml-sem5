# 1. Import necessary libraries
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler

# 2. Load and normalize the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape data from 28x28 images to 784-dimensional vectors
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# 3. The dataset is already split by the loader (60k training, 10k testing)

# 4. Preprocess features with StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 5. Define network configurations
network_depths = [1, 5, 10, 20]
results = {}

# 6 & 7. Loop through configurations to train and evaluate models
for depth in network_depths:
    print(f"\n--- Training Model with {depth} Hidden Layer(s) ---")

    # Define the model
    model = Sequential()
    model.add(Input(shape=(784,))) # Input layer
    # Add hidden layers
    for _ in range(depth):
        model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='softmax')) # Output layer

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Record training time
    start_time = time.time()
    history = model.fit(x_train_scaled, y_train,
                        epochs=25,
                        batch_size=128,
                        verbose=0) # verbose=0 silences the epoch-by-epoch output
    end_time = time.time()
    training_time = end_time - start_time

    # Evaluate the model
    train_loss, train_acc = model.evaluate(x_train_scaled, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(x_test_scaled, y_test, verbose=0)

    # Store results
    results[depth] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'history': history,
        'time': training_time
    }

    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

# 8. Plot the results

# a) Plot training loss curves
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
for depth, data in results.items():
    plt.plot(data['history'].history['loss'], label=f'{depth} hidden layers')

plt.title('Training Loss vs. Epochs for Different Network Depths', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Training Loss (Sparse Categorical Crossentropy)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# b) Create a bar chart for accuracies
labels = [f'{d} Layers' for d in network_depths]
train_accuracies = [res['train_acc'] for res in results.values()]
test_accuracies = [res['test_acc'] for res in results.values()]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, train_accuracies, width, label='Training Accuracy')
rects2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy')

# Add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy')
ax.set_title('Training vs. Test Accuracy by Network Depth')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.bar_label(rects1, padding=3, fmt='%.3f')
ax.bar_label(rects2, padding=3, fmt='%.3f')
ax.set_ylim(0, 1.1)

fig.tight_layout()
plt.show()