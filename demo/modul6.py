import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load dataset CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Definisi model CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation="softmax")   # Softmax → probabilitas
])

# Compile
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Training
history = model.fit(
    train_images, train_labels,
    epochs=5,   # bisa diperbesar misalnya 10 atau 20
    validation_data=(test_images, test_labels)
)

# Plot hasil training
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

# -----------------------------
# Evaluasi di test set
# -----------------------------
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# -----------------------------
# Prediksi gambar baru
# -----------------------------
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# Ambil 5 sampel dari test set
sample_images = test_images[:5]
sample_labels = test_labels[:5]

predictions = model.predict(sample_images)

plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(sample_images[i])
    pred_label = np.argmax(predictions[i])
    true_label = sample_labels[i][0]
    color = "green" if pred_label == true_label else "red"
    plt.xlabel(f"{class_names[pred_label]}\n(True: {class_names[true_label]})", color=color)
plt.show()
