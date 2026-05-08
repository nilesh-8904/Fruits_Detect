import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    r'dataset/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    r'dataset/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(36, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=25,

    # ---------------
    epochs=25,
    # -----------------
    validation_data=validation_generator,
    validation_steps=10
)

# Save the model
model.save('Model.h5')

# Save the class labels
labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())

with open('label.txt', 'w') as f:
    for key in sorted(labels.keys()):
        f.write(f"{key}: {labels[key]}\n")
