import pathlib
import numpy as np
from helper_functions import plot_loss_curves

# Load data
train_dir = pathlib.Path("train/")  # turn our training path into a Python path
class_names = np.array(
    sorted([item.name for item in train_dir.glob('*')]))  # created a list of class_names from the subdirectories
print(class_names)

# Trainingsdaten vorbereiten
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing

# Trainingsdaten vorbereiten
train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                           validation_split=0.2,
                                                                           subset='training',
                                                                           seed=42,
                                                                           image_size=(224,224),
                                                                           batch_size=32,
                                                                           label_mode='categorical',
                                                                           shuffle=True)

val_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                                validation_split=0.2,
                                                                                subset='validation',
                                                                                seed=42,
                                                                                image_size=(224, 224),
                                                                                batch_size=32,
                                                                                label_mode='categorical',
                                                                                shuffle=True)

# Build data augmentation layer
data_augmentation = Sequential([
  preprocessing.RandomFlip('horizontal'),
  preprocessing.RandomHeight(0.2),
  preprocessing.RandomWidth(0.2),
  preprocessing.RandomZoom(0.2),
  preprocessing.RandomRotation(0.2),
  preprocessing.Rescaling(1./255)
], name="data_augmentation")

# 1. Create base model with tf.keras.applications
base_model = tf.keras.applications.resnet50.ResNet50(include_top=False)

# 2. Freeze the base model (so the pre-learned patterns remain)
base_model.trainable = True

for layer in base_model.layers[:10]:
    layer.trainable = False

# 3. Create inputs into the base model
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
x = data_augmentation(inputs)

# 5. Pass the inputs to the base_model (note: using tf.keras.applications, EfficientNet inputs don't have to be normalized)
x = base_model(inputs)
# Check data shape after passing it to base_model
print(f"Shape after base_model: {x.shape}")

# 6. Average pool the outputs of the base model (aggregate all the most important information, reduce number of computations)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
print(f"After GlobalAveragePooling2D(): {x.shape}")

# 7. Create the output activation layer
outputs = tf.keras.layers.Dense(2, activation="sigmoid", name="output_layer")(x)

# 8. Combine the inputs with the outputs into a model
model3 = tf.keras.Model(inputs, outputs)

# 9. Compile the model
model3.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# 10. Fit the model (we use less steps for validation so it's faster)
history3 = model3.fit(train_data,
                                 epochs=5,
                                 steps_per_epoch=len(train_data),
                                 validation_data=val_data,
                                 # Go through less of the validation data so epochs are faster (we want faster experiments!)
                                 validation_steps=len(val_data))

#Save the model
np.save('Auswertung/history_model_3.npy', history3.history)
model3.save('Auswertung/model3')


# Plot the training curves
plot_loss_curves(history3)