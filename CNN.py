from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from numpy import random
import tensorflow as tf
from helper_functions import view_random_image, walk_through_dir, plot_loss_curves

# Verzeichnisse anschauen
walk_through_dir("train")

# Get the class names (programmatically, this is much more helpful with a longer list of classes)
import pathlib
import numpy as np

train_dir = pathlib.Path("train/")  # turn our training path into a Python path
class_names = np.array(
    sorted([item.name for item in train_dir.glob('*')]))  # created a list of class_names from the subdirectories
print(class_names)

# Zeige ein zufälliges Bild jeder Klasse
img = view_random_image(target_dir="train/",
                        target_class="cats")

img = view_random_image(target_dir="train/",
                        target_class="dogs")

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
                                                                   class_mode='binary',
                                                                   shuffle=True,
                                                                   subset='training')

validation_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                        target_size=(224, 224),
                                                                        batch_size=32,
                                                                        class_mode='binary',
                                                                        shuffle=True,
                                                                        subset='validation')

# Visualisieren
augmented_images, augmented_labels = train_data_augmented.next()
# Show original image and augmented image
random_number = random.randint(0, 32)  # we're making batches of size 32, so we'll get a random instance
img = augmented_images[random_number]
plt.imshow(img)
plt.title(f"Augmented image")
plt.axis(False);
plt.show();
print(f"Image shape: {img.shape}")

# Model erzeugen
model_1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),  # same input shape as our images
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
 #   tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
  #  tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=3, min_lr=0.001)

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

checkpoint_filepath = '/Auswertung/checkpoint_model_1'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Fit the model
history_1 = model_1.fit(train_data_augmented,
                        epochs=8,
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=validation_data_augmented,
                        validation_steps=len(validation_data_augmented),
                        callbacks=[reduce_lr, earlyStopping])

np.save('Auswertung/history_model_dropout.npy', history_1.history)
model_1.save('Auswertung/model1_dropout')

# Plot the training curves
plot_loss_curves(history_1)

