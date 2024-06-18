import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount('/content/gdrive')

# Define the path to your dataset
train_dir = '/content/gdrive/My Drive/dataset/train'
test_dir = '/content/gdrive/My Drive/dataset/test'

import os
# Get the class labels from the folder names in the train directory
train_labels = sorted(os.listdir(train_dir))

# Get the class labels from the folder names in the test directory
test_labels = sorted(os.listdir(test_dir))

# Print the class labels
print('Train Class Labels:', train_labels)
print('Test Class Labels:', test_labels)

# Define the class labels in the desired order
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf', 'Leaf Blast']

# Set the image size and batch size
image_size = (800, 800)
batch_size = 32

# Data augmentation and normalization for training dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Randomly rotate images
    horizontal_flip=True)  # Randomly flip images horizontally

# Load the training dataset
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    classes=class_names)

# Normalization for testing dataset (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)
# Load the testing dataset
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    classes=class_names)

# Print the train labels
train_labels = train_generator.classes
print("Train Labels:")
print(train_labels)

# Print the test labels
test_labels = test_generator.classes
print("Test Labels:")
print(test_labels)

# Visualize sample images from each class
num_samples_per_class = 3
# Retrieve a batch of images and labels from the test generator
images, labels = next(test_generator)
# Plot sample images
fig, axes = plt.subplots(len(class_names), num_samples_per_class, figsize=(10, 10))

for i, ax in enumerate(axes):
    class_images = images[np.argmax(labels, axis=1) == i]
    class_name = class_names[i]

    num_samples = min(num_samples_per_class, class_images.shape[0])

    for j in range(num_samples):
        # Display sample images
        ax[j].imshow(class_images[j])
        ax[j].axis('off')
        ax[j].set_title(class_name)

    for j in range(num_samples, num_samples_per_class):
        # Remove unused subplots
        ax[j].remove()

plt.tight_layout()
plt.show()

# Define the CNN model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(800, 800, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # Assuming 4 disease classes
])

model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=40)

