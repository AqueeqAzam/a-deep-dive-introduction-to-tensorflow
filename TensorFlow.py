# TensorFlow Basics & Use Case: House Price Prediction
import tensorflow as tf
import numpy as np

print("TensorFlow Version:", tf.__version__)

# ------------------------
# 1. Tensor Creation
# ------------------------

scalar = tf.constant(7)
vector = tf.constant([10, 20, 30])
matrix = tf.constant([[1, 2], [3, 4]])
tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print("\nScalar:", scalar.numpy())
print("Vector:", vector.numpy())
print("Matrix:\n", matrix.numpy())
print("Tensor shape:", tensor.shape)

# ------------------------
# 2. Variable & Assign
# ------------------------

change_tensor = tf.Variable([10, 20])
change_tensor[0].assign(99)
print("\nUpdated Variable:", change_tensor.numpy())

# ------------------------
# 3. Random & Zeros/Ones
# ------------------------

rand_tensor = tf.random.normal(shape=(2, 3), seed=42)
zeros_tensor = tf.zeros([2, 2])
ones_tensor = tf.ones([2, 2])
range_tensor = tf.range(1, 5)

print("\nRandom:\n", rand_tensor.numpy())
print("Zeros:\n", zeros_tensor.numpy())
print("Range:\n", range_tensor.numpy())

# ------------------------
# 4. Tensor Properties
# ------------------------

t = tf.constant(np.random.randint(0, 10, size=(3, 3)))
print("\nTensor:\n", t.numpy())
print("Shape:", t.shape)
print("Rank:", t.ndim)
print("Size:", tf.size(t).numpy())

# ------------------------
# 5. Indexing
# ------------------------

sample = tf.constant([1, 2, 3, 4, 5])
print("\nFirst 2:", sample[:2].numpy())
print("Last 2:", sample[-2:].numpy())
print("Reversed:", sample[::-1].numpy())

# ------------------------
# 6. Basic Math
# ------------------------

x = tf.constant([1.0, 2.0, 3.0])
print("\nAdd 5:", (x + 5).numpy())
print("Square root:", tf.sqrt(x).numpy())
print("Exp:", tf.exp(x).numpy())

# ------------------------
# 7. Stats
# ------------------------

data = tf.constant([10, 20, 30, 40], dtype=tf.float32)
print("\nMean:", tf.reduce_mean(data).numpy())
print("Std Dev:", tf.math.reduce_std(data).numpy())
print("Variance:", tf.math.reduce_variance(data).numpy())

# ------------------------
# 8. One-Hot Encoding
# ------------------------

categories = [0, 1, 2]
print("\nOne Hot Encoding:\n", tf.one_hot(categories, depth=3).numpy())

# ------------------------
# 9. Tensor ↔ NumPy
# ------------------------

tensor_np = tf.constant([1, 2, 3])
np_arr = tensor_np.numpy()
print("\nTensor to NumPy:", np_arr)

# ------------------------
# 🎯 10. Mini Use Case: Predict House Price (Linear Model)
# ------------------------

# Features: Area (sqft)
X = tf.constant([[1000.0], [1500.0], [2000.0], [2500.0]], dtype=tf.float32)

# Target: Price ($1000s)
y = tf.constant([[100.0], [150.0], [200.0], [250.0]], dtype=tf.float32)

# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile Model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train Model
print("\nTraining model...")
model.fit(X, y, epochs=100, verbose=0)

# Predict
area = tf.constant([[3000.0]])
prediction = model.predict(area)
print(f"\nPredicted price for 3000 sqft: ${prediction[0][0]:.2f}k")
