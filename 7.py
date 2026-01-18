import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to 1D
y_train = y_train.flatten()
y_test = y_test.flatten()

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
print("🚀 Training CNN on CIFAR-10 dataset...")
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

# Evaluate
print("\n🔍 Evaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

print(f"\n✅ Test accuracy: {test_acc * 100:.2f}%")
