import cv2
import time
from modules.models import cargar_modelo_clasificacion, cargar_modelo_yolo
from modules.camera_utils import seleccionar_camara
from modules.detector import procesar_deteccion

# === CARGAR MODELOS ===
interpreter = cargar_modelo_clasificacion()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
modelo_yolo = cargar_modelo_yolo()

# === INICIAR CÁMARA ===
indice = seleccionar_camara()
cap = cv2.VideoCapture(indice)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("📷 Cámara activada. Presiona 'q' para salir.")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    results = modelo_yolo(frame)[0]
    output = procesar_deteccion(frame, results, interpreter, input_details, output_details)

    fps = 1.0 / (time.time() - start_time)
    cv2.putText(output, f"FPS: {fps:.1f}", (10, output.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

    cv2.imshow("LevelWater - Botella segmentada", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()