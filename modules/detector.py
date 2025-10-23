import numpy as np
import cv2
from .classifier import clasificar_botella
import random

LABEL_MAP_ES_EN = {
    "derramado": "SPILLED",
    "lleno": "FULL",
    "medio": "HALF",
    "vacio": "EMPTY"
}

def procesar_deteccion(frame, results, interpreter, input_details, output_details):
    output = np.ones_like(frame) * 255
    botella_detectada = False
    resultado_final = None

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        nombre = results.names[cls_id]
        conf = float(box.conf[0])

        if nombre == "bottle" and conf > 0.5:
            botella_detectada = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if results.masks is not None:
                mask = results.masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask = (mask > 0.5).astype(np.uint8) * 255

                # Contorno negro
                contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output, contornos, -1, (0, 0, 0), 2)

                # Azul en parte inferior
                mask_roi = np.zeros_like(mask)
                nivel_y = int(y1 + (y2 - y1) * 0.4)
                mask_roi[nivel_y:y2, x1:x2] = mask[nivel_y:y2, x1:x2]
                blue = np.zeros_like(frame)
                blue[:] = (255, 0, 0)
                output = np.where(mask_roi[:, :, None] == 255, blue, output)

                # Clasificación
                resultado = clasificar_botella(frame, x1, y1, x2, y2, interpreter, input_details, output_details)
                if resultado:
                    etiqueta_es, confianza = resultado  # etiqueta en ES

                    # Rangos estimados por clase (en ES)
                    rango_llenado = {
                        "derramado": (104, 110),
                        "lleno": (95, 100),
                        "medio": (45, 55),
                        "vacio": (0, 10)
                    }

                    # Valor aleatorio dentro del rango
                    rango = rango_llenado.get(etiqueta_es, (0, 100))
                    porcentaje_llenado = random.uniform(*rango)

                    # Traducir etiqueta a EN antes de retornar
                    etiqueta_en = LABEL_MAP_ES_EN.get(etiqueta_es, "UNKNOWN")

                    # Devolver en EN: (label_en, nivel, confianza)
                    resultado_final = (etiqueta_en, porcentaje_llenado, confianza)

    return output, resultado_final