import numpy as np
import cv2
from .classifier import clasificar_botella

def procesar_deteccion(frame, results, interpreter, input_details, output_details):
    output = np.ones_like(frame) * 255
    botella_detectada = False

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
                    etiqueta, porcentaje = resultado
                    texto = f"{etiqueta.upper()} ({porcentaje:.1f}%)"
                    cv2.putText(output, texto, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if not botella_detectada:
        cv2.putText(output, "Botella no detectada", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

    return output