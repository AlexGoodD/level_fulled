from bing_image_downloader import downloader

# Etiquetas que NO deben contener botellas
etiquetas = ["laptop", "tree", "dog", "sofa", "person", "car", "phone"]

# Crear carpeta base
for etiqueta in etiquetas:
    print(f"Descargando: {etiqueta}")
    downloader.download(etiqueta, limit=20, output_dir='sin_botella', adult_filter_off=True, force_replace=False, timeout=60)

print("✅ Descarga completada.")