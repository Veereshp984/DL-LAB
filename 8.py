import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
max_words = 10000
max_len = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# Preprocessing (padding sequences)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Build RNN model
model = models.Sequential([
    layers.Embedding(max_words, 64, input_length=max_len),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
print("🚀 Training RNN on IMDB dataset...")
model.fit(x_train, y_train, epochs=3,
          batch_size=64, validation_split=0.2, verbose=1)

# Evaluate
print("\n🔍 Evaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
