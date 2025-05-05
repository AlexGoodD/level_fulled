import os
import sys
import tensorflow as tf
from ultralytics import YOLO
from .constants import MODELO_TFLITE_PATH, YOLO_MODEL

def cargar_modelo_clasificacion():
    if not os.path.isfile(MODELO_TFLITE_PATH):
        print("❌ Modelo TFLite no encontrado.")
        sys.exit()
    try:
        interpreter = tf.lite.Interpreter(model_path=MODELO_TFLITE_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"❌ Error al cargar modelo TFLite: {e}")
        sys.exit()

def cargar_modelo_yolo():
    return YOLO(YOLO_MODEL)