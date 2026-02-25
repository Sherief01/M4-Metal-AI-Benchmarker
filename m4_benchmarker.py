import tensorflow as tf
import time
import os

# Set logging to clean up output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("\n" + "="*50)
print("🚀 M4 APPLE SILICON AI BENCHMARKER")
print("="*50)

# 1. Hardware Detection
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"✅ Found GPU: {gpu.name}")
    print("✨ Metal Acceleration: ENABLED")
else:
    print("⚠️ Warning: GPU not detected. Ensure tensorflow-metal is installed.")

# 2. Data Preparation (MNIST)
print("\n📦 Loading Dataset (MNIST)...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Model Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Training Benchmark
print("\n🔥 Starting High-Intensity Training on M4 GPU...")
start_time = time.time()

# Force training on GPU:0 (M4 Metal)
with tf.device('/GPU:0'):
    model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1)

duration = time.time() - start_time
print("\n" + "="*50)
print(f"🏁 Benchmark Completed in: {duration:.2f} seconds")
print(f"🖥️ Platform: Mac Mini M4 (ARM64)")
print("="*50 + "\n")

# Save performance result to a file for GitHub
with open("benchmark_result.txt", "w") as f:
    f.write(f"M4 GPU Training Time (5 Epochs): {duration:.2f}s")
