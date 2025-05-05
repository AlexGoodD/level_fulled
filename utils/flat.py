import os
import shutil

# Carpeta donde están las subcarpetas de imágenes descargadas
carpeta_base = "sin_botella"
carpeta_destino = "sin_botella_flat"

# Crear carpeta destino si no existe
os.makedirs(carpeta_destino, exist_ok=True)

contador = 0  # Para evitar nombres duplicados

for subcarpeta in os.listdir(carpeta_base):
    subcarpeta_path = os.path.join(carpeta_base, subcarpeta)
    
    if os.path.isdir(subcarpeta_path):
        for archivo in os.listdir(subcarpeta_path):
            origen = os.path.join(subcarpeta_path, archivo)

            # Verifica que sea un archivo (evita carpetas ocultas, etc.)
            if os.path.isfile(origen):
                # Genera nombre único para cada imagen
                nuevo_nombre = f"{subcarpeta}_{contador}{os.path.splitext(archivo)[1]}"
                destino = os.path.join(carpeta_destino, nuevo_nombre)

                shutil.copy(origen, destino)
                contador += 1

print(f"✅ {contador} imágenes copiadas a '{carpeta_destino}'")