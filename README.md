# LevelFulled

> Sistema de visión por computadora para **detección y clasificación de niveles de llenado en recipientes industriales** en tiempo real, usando una pipeline de dos etapas: detección con MobileNet-SSD + clasificación con un modelo TFLite entrenado a medida.

---

## Problema que resuelve

En líneas de producción industrial, verificar manualmente el nivel de llenado de recipientes es lento, costoso y propenso a errores humanos. Este sistema automatiza esa inspección usando la cámara existente de la línea, **sin hardware adicional**.

---

## Demo

<!-- Editar contenido -->
![Demo](docs/demo.gif)

| Vacío | Medio | Lleno |
|:-----:|:-----:|:-----:|
| ![empty](docs/empty.jpg) | ![half](docs/half.jpg) | ![full](docs/full.jpg) |

---

## Arquitectura

```
Cámara en tiempo real
        ↓
MobileNet-SSD  (detección)
→ Localiza el recipiente en el frame
→ Extrae bounding box
        ↓
Modelo TFLite custom  (clasificación)
→ Clasifica nivel: [ vacío | medio | lleno ]
        ↓
Output: coordenadas + etiqueta + confianza
```

### ¿Por qué dos modelos en lugar de uno?

MobileNet-SSD está optimizado para **localización rápida**. Usar un clasificador separado y ligero (TFLite) sobre el ROI detectado permite:

- Mayor precisión en la clasificación sin sacrificar velocidad
- Re-entrenar solo la clasificación cuando cambia el tipo de recipiente
- Inferencia eficiente en CPU sin GPU dedicada

---

## Resultados

| Métrica | Valor |
|---------|-------|
| Precisión clasificación (test set) | **76%** |
| FPS promedio (CPU) | **~15 fps** |
| Clases soportadas | vacío / medio / lleno |
| Imágenes en dataset de entrenamiento | **750 imágenes** |

> Entrenado y validado con imágenes de recipientes de [describe el contexto: laboratorio / línea de producción / simulado].

---

## Stack técnico

| Componente | Tecnología |
|-----------|------------|
| Detección de objetos | MobileNet-SSD (Caffe) |
| Clasificación de nivel | TFLite (modelo custom) |
| Visión por computadora | OpenCV |
| Runtime | Python 3.x + TensorFlow |

---

## Instalación

```bash
git clone https://github.com/AlexGoodD/LevelFulled
cd LevelFulled
pip install -r requirements.txt
```

Coloca los modelos preentrenados en `modelos/`:

```
modelos/
├── MobileNetSSD_deploy.caffemodel
├── MobileNetSSD_deploy.prototxt
└── modelo_nivel_llenado.tflite
```

---

## Uso

```bash
python main.py
```

> La primera ejecución puede solicitar permisos de cámara y cerrarse. Vuelve a correrlo después de aceptar el permiso.

---

## Entrenar el modelo de clasificación

1. Prepara tu dataset en la ruta configurada en el script de entrenamiento
2. Entrena el modelo
3. Comprime y convierte a TFLite antes de usar:

```bash
python utils/comprension.py
```

Esto convierte el modelo a formato TFLite y reduce su tamaño para inferencia eficiente en producción.

---

## Estructura del proyecto

```
LevelFulled/
├── main.py                  # Entry point — cámara + pipeline completa
├── modelos/                 # Modelos de detección y clasificación
│   ├── MobileNetSSD_deploy.caffemodel
│   ├── MobileNetSSD_deploy.prototxt
│   └── modelo_nivel_llenado.tflite
├── utils/
│   ├── comprension.py       # Conversión y compresión a TFLite
│   └── image_utils.py       # Utilidades para manejo de imágenes
├── docs/                    # Assets para README (demo, capturas)
└── requirements.txt
```

---

## Referencias

- [MobileNet-SSD — Wei Liu et al. (2015)](https://arxiv.org/abs/1512.02325)
- [TensorFlow Lite — Model Optimization Guide](https://www.tensorflow.org/lite/performance/model_optimization)
- [OpenCV — Object Detection](https://docs.opencv.org/4.x/d2/d64/tutorial_table_of_content_objdetect.html)

---

## Autor

**Alex** — [@AlexGoodD](https://github.com/AlexGoodD)
