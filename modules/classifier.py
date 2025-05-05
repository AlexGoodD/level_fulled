import numpy as np
import cv2
from .constants import CLASES_NIVEL

def clasificar_botella(frame, x1, y1, x2, y2, interpreter, input_details, output_details):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    img = cv2.resize(roi, (224, 224))
    img_array = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = np.argmax(pred)
    return CLASES_NIVEL[idx], pred[idx] * 100