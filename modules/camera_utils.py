import cv2
import sys

def detectar_camaras(max_index=5):
    disponibles = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            disponibles.append(i)
        cap.release()
    return disponibles

def seleccionar_camara():
    camaras = detectar_camaras()
    if not camaras:
        print("❌ No se encontraron cámaras.")
        sys.exit()
    print("\n🎥 Cámaras disponibles:")
    for i, cam in enumerate(camaras):
        print(f"[{i}] Cámara {cam}")
    while True:
        try:
            seleccion = int(input("Selecciona una cámara por número: "))
            if 0 <= seleccion < len(camaras):
                return camaras[seleccion]
        except ValueError:
            print("❌ Entrada inválida.")