import pathlib
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from helper_functions import create_model, plot_loss_curves
import tensorflow as tf

# Load data
train_dir = pathlib.Path("train/")  # turn our training path into a Python path
class_names = np.array(
    sorted([item.name for item in train_dir.glob('*')]))  # created a list of class_names from the subdirectories
print(class_names)

# Trainingsdaten vorbereiten
train_datagen_augmented = ImageDataGenerator(rescale=1 / 255.,
                                             validation_split=0.2,
                                             rotation_range=0.2,  # rotate the image slightly
                                             shear_range=0.2,  # shear the image
                                             zoom_range=0.2,  # zoom into the image
                                             width_shift_range=0.2,  # shift the image width ways
                                             height_shift_range=0.2,  # shift the image height ways
                                             horizontal_flip=True)  # flip the image on the horizontal axis

train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                   target_size=(224, 224),
                                                                   batch_size=32,
                                                                   class_mode='categorical',
                                                                   shuffle=True,
                                                                   subset='training')

validation_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                        target_size=(224, 224),
                                                                        batch_size=32,
                                                                        class_mode='categorical',
                                                                        shuffle=True,
                                                                        subset='validation')

# Resnet 50 V2 feature vector
resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"

# EfficientNet0 feature vector
efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

model2 = create_model(resnet_url, 2)

model2.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

history2 = model2.fit(train_data_augmented,
                      epochs=4,
                      steps_per_epoch=len(train_data_augmented),
                      validation_data=validation_data_augmented,
                      validation_steps=len(validation_data_augmented))

np.save('Auswertung/history_model_2.npy', history2.history)
model2.save('Auswertung/model2')

# Plot the training curves
plot_loss_curves(history2)