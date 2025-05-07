import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# === CONFIGURACIÓN ===
dataset_dir = "/Users/alejandro/Desktop/LevelWater/dataset"
img_size = (224, 224)
batch_size = 32
num_classes = 4

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

# === GENERADORES DE DATOS ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = val_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === MODELO CON TRANSFER LEARNING + FINE-TUNING ===
base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Paso 1: congelado

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === ENTRENAMIENTO INICIAL (con base congelada) ===
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

print("🚀 Entrenando modelo base...")
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=[early_stop]
)

# === FINE-TUNING (descongelar últimas capas) ===
print("🔧 Activando fine-tuning...")
base_model.trainable = True
# Opcional: congelar las capas más profundas si es necesario
for layer in base_model.layers[:-30]:  # Ajusta según tu hardware/datos
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=[early_stop]
)

# === GRAFICAR PRECISIÓN TOTAL ===
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Validación')
plt.legend()
plt.title("Precisión del modelo con fine-tuning")
plt.show()

# === GUARDAR MODELO ===
model_path = "/Users/alejandro/Desktop/LevelWater/modelos/modelo_finetuned_mobilenet.h5"
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

with open("/Users/alejandro/Desktop/LevelWater/modelos/modelo_finetuned_mobilenet.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Modelo convertido y guardado como .tflite")