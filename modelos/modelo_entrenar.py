import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# === RUTA AL DATASET ===
dataset_dir = "/Users/alejandro/Desktop/LevelWater/dataset"  # Ajustar ruta es distinta

# === LIMPIAR IMÁGENES INVÁLIDAS ===
def limpiar_dataset(path):
    extensiones_validas = (".jpg", ".jpeg", ".png")
    for root, _, files in os.walk(path):
        for f in files:
            archivo = os.path.join(root, f)
            if not f.lower().endswith(extensiones_validas):
                print(f"❌ Eliminando archivo no válido: {archivo}")
                os.remove(archivo)
                continue
            try:
                img = Image.open(archivo)
                img.verify()
            except Exception:
                print(f"⚠️ Eliminando imagen corrupta: {archivo}")
                os.remove(archivo)

print("🧹 Limpiando imágenes inválidas...")
limpiar_dataset(dataset_dir)
print("✅ Limpieza completa.\n")

# === CARGAR DATOS PARA ENTRENAMIENTO ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === DEFINIR MODELO CNN ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # Asegúrate de tener 4 clases
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# === ENTRENAR MODELO ===
history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=[early_stop]
)

# === GRAFICAR PRECISIÓN ===
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.legend()
plt.title("Precisión del modelo")
plt.show()

# === GUARDAR MODELO ===
model_path = "/Users/alejandro/Desktop/LevelWater/modelos/modelo_nivel_llenado.h5"
model.save(model_path)
print(f"✅ Modelo guardado en: {model_path}")

# === MATRIZ DE CONFUSIÓN ===
val_gen.reset()
y_true = val_gen.classes
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)

class_names = list(val_gen.class_indices.keys())

matriz = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.show()

# === CONVERSIÓN A TFLITE ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("/Users/alejandro/Desktop/LevelWater/modelos/modelo_nivel_llenado.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Modelo convertido y guardado como .tflite")