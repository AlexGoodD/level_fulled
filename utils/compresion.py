import tensorflow as tf

# Cargar el modelo original
modelo = tf.keras.models.load_model("modelos/modelo_nivel_llenado.h5")

# Convertir el modelo a un formato más ligero con cuantización
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Guardar el modelo comprimido
with open("modelos/modelo_nivel_llenado.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo comprimido guardado como modelo_nivel_llenado.tflite")