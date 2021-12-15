# PREDICTION
# Hier wird nochmal alles geladen, damit man es auch einzeln ausführen kann
from helper_functions import pred_and_plot, load_and_prep_image
from tensorflow import keras
import pathlib
import numpy as np

train_dir = pathlib.Path("train/")  # turn our training path into a Python path
class_names = np.array(
    sorted([item.name for item in train_dir.glob('*')]))  # created a list of class_names from the subdirectories
model = keras.models.load_model('Auswertung/model1_dropout')

pred_and_plot(model, "test/9.jpg", class_names, True, 224)
