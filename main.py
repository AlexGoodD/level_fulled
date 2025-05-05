import cv2
import numpy as np
import tensorflow as tf
import os
import sys

# === RUTAS AL MODELO DE DETECCIÓN ===
prototxt = "modelos/MobileNetSSD_deploy.prototxt"
modelo_caffe = "modelos/MobileNetSSD_deploy.caffemodel"

# === RUTA AL MODELO DE CLASIFICACIÓN ===
modelo_nivel = "modelos/modelo_nivel_llenado.tflite"

# === VERIFICAR ARCHIVOS ===
if not os.path.isfile(prototxt) or not os.path.isfile(modelo_caffe):
    print("❌ No se encontraron los archivos del modelo Caffe.")
    sys.exit()

if not os.path.isfile(modelo_nivel):
    print("❌ No se encontró el modelo .tflite de clasificación.")
    sys.exit()

# === CARGAR MODELO DE CLASIFICACIÓN (TFLITE) ===
try:
    interpreter = tf.lite.Interpreter(model_path=modelo_nivel)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"❌ Error al cargar el modelo TFLite: {e}")
    sys.exit()

# Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === CLASES DE CLASIFICACIÓN ===
clases_nivel = ['derramado', 'lleno', 'medio', 'vacio']

# === CLASES DE DETECCIÓN DEL SSD ===
clases_ssd = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
              "sofa", "train", "tvmonitor"]

# === CARGAR MODELO DE DETECCIÓN ===
net = cv2.dnn.readNetFromCaffe(prototxt, modelo_caffe)

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    sys.exit()

print("📷 Cámara activada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer el frame de la cámara.")
        break

    alto, ancho = frame.shape[:2]

    # === DETECCIÓN DE OBJETOS ===
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    botella_detectada = False

    for i in range(detections.shape[2]):
        confianza = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if class_id < len(clases_ssd) and clases_ssd[class_id] == "bottle" and confianza > 0.5:
            botella_detectada = True

            # Obtener coordenadas y corregir límites
            box = detections[0, 0, i, 3:7] * np.array([ancho, alto, ancho, alto])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(ancho, endX), min(alto, endY)

            botella_roi = frame[startY:endY, startX:endX]

            if botella_roi.size > 0:
                try:
                    img = cv2.resize(botella_roi, (224, 224))
                    img_array = np.expand_dims(img / 255.0, axis=0).astype(np.float32)

                    # Clasificación
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]['index'])[0]
                    idx = np.argmax(pred)
                    etiqueta = clases_nivel[idx]
                    conf = pred[idx] * 100

                    # Dibujar
                    texto = f"{etiqueta.upper()} ({conf:.1f}%)"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, texto, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                except Exception as e:
                    print(f"⚠️ Error al procesar ROI: {e}")

    if not botella_detectada:
        cv2.putText(frame, "🚫 SIN BOTELLA DETECTADA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow("Detección + Clasificación", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()