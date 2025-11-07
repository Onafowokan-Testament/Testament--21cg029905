# !pip install -q kaggle
# !mkdir -p ~/.kaggle



# from google.colab import files
# files.upload()


# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d msambare/fer2013
# !unzip -q fer2013.zip -d fer2013



import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# STEP 2: Define dataset paths
train_dir = '/content/fer2013/train'
test_dir = '/content/fer2013/test'

# STEP 3: Preprocess the images
# Rescale pixel values to [0, 1] and apply simple data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# STEP 4: Load images directly from the folder
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),   # FER2013 images are 48x48
    batch_size=64,
    color_mode='grayscale', # images are black and white
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

# STEP 5: Print class labels
print("Class labels:", train_generator.class_indices)

# model_training.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
train_dir = 'fer2013/train'
test_dir = 'fer2013/test'

# Image generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    epochs=35,
    validation_data=test_generator
)

# Save model
model.save('face_emotionModel.h5')

print("✅ Model trained and saved successfully as face_emotionModel.h5")