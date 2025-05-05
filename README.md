# LevelFulled

LevelFulled es un proyecto diseñado para detectar objetos y clasificar el nivel de llenado de recipientes utilizando modelos de aprendizaje profundo. Este proyecto combina un modelo de detección basado en MobileNet-SSD y un modelo de clasificación TFLite para realizar estas tareas.

## Requisitos

- Python 3.x
- OpenCV
- TensorFlow
- NumPy

## Instalación

1. Clona este repositorio en tu máquina local.
2. Instala las dependencias necesarias ejecutando:

```bash
   pip install -r requirements.txt
```

3. Asegúrate de que los modelos necesarios se encuentren en la carpeta modelos/

```bash
   MobileNetSSD_deploy.caffemodel
   MobileNetSSD_deploy.prototxt
   modelo_nivel_llenado.tflite
   ```
## Uso
1. Ejecuta el archivo principal del proyecto

```bash
python main.py
```

2. El programa iniciará la cámara y comenzará a detectar objetos y clasificar el nivel de llenado de recipientes (es probable que la primera vez el programa te pida permiso para acceder a la cámara y después se cierre, solo vuelvelo a correr)

## Estructura del proyecto

- main.py: Archivo principal que ejecuta el programa
- modelos/: Carpeta que contiene los modelos de detección y clasificación.
- Utils: Scripts auxiliares para tareas como compresión de modelos, y manejo de imágenes.

## Créditos
- El modelo de detección MobileNet-SSD fue desarrollado por [Wei Liu et al](https://arxiv.org/abs/1512.02325)
- El modelo de clasificación fue desarrollado especialmente para este proyecto


