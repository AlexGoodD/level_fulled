import cv2
import time
from PIL import Image, ImageTk
import customtkinter as ctk

# === MODULOS DE CAMARAS ===
from modules.models import cargar_modelo_clasificacion, cargar_modelo_yolo
from modules.camera_utils import seleccionar_camara
from modules.detector import procesar_deteccion

# === MODULOS DE UI ===
from modules.ui.topbar import create_topbar
from modules.ui.cards import create_card_view
from modules.ui.stats import create_stat_card_view

# === CONFIGURACIÓN GENERAL ===
interpreter = cargar_modelo_clasificacion()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
modelo_yolo = cargar_modelo_yolo()

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("AquaSight - Level Sense AI Detector")
app.geometry("1300x800")
app.configure(fg_color="#FAFAFA")

# --- TOPBAR ---
create_topbar(app, realtime_mode=True)

# ---------- CONTENEDOR PRINCIPAL ----------
main_frame = ctk.CTkFrame(app, fg_color="transparent")
main_frame.pack(pady=40)

# Asegurar columnas uniformes
main_frame.grid_columnconfigure((0, 1), weight=1, uniform="cards")


# --- TARJETA 1: Real-Time Camera View ---
shadow_view_1, real_view, camera_content = create_card_view(
    main_frame,
    icon="assets/camera_icon.png",
    title="Camera View",
    bar_color="#0BADAC"
)
shadow_view_1.grid(row=0, column=0, padx=50, pady=0)
real_view.grid(row=0, column=0, padx=50, pady=0)

# --- TARJETA 2: AI Processed View ---
shadow_view_2, ai_view, ai_content = create_card_view(
    main_frame,
    icon="assets/ai_icon.png",
    title="AI View",
    bar_color="#F76624"
)
shadow_view_2.grid(row=0, column=1, padx=50, pady=0)
ai_view.grid(row=0, column=1, padx=50, pady=0)

# --- ETIQUETAS PARA MOSTRAR LAS CÁMARAS ---
camera_label = ctk.CTkLabel(camera_content, text="")
camera_label.pack(expand=True, fill="both")

ai_label = ctk.CTkLabel(ai_content, text="")
ai_label.pack(expand=True, fill="both")

# ---------- SECCIÓN DE ESTADÍSTICAS ----------
stats_frame = ctk.CTkFrame(app, fg_color="transparent")
stats_frame.pack(pady=40)

# Asegurar columnas uniformes
stats_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="stats")

# --- TARJETA DE ESTADÍSTICA 1 ---
shadow_s1, stat1, stat1_value_label, stat1_sublabel = create_stat_card_view(
    stats_frame,
    value="0%",
    label="Fill Level",
    sublabel="DESCONOCIDO",
    text_color="#0BADAC",
    icon_path="assets/fill_icon.png"
)
shadow_s1.grid(row=0, column=0, padx=40)
stat1.grid(row=0, column=0, padx=40)

# --- TARJETA DE ESTADÍSTICA 2 ---
shadow_s2, stat2, stat2_value_label, stat2_sublabel = create_stat_card_view(
    stats_frame,
    value="0%",
    label="Confidence",
    sublabel="DESCONOCIDO",
    text_color="#F76624",
    icon_path="assets/check_icon.png"
)
shadow_s2.grid(row=0, column=1, padx=40)
stat2.grid(row=0, column=1, padx=40)

# --- TARJETA DE ESTADÍSTICA 3 ---
shadow_s3, stat3, stat3_value_label, stat3_sublabel = create_stat_card_view(
    stats_frame,
    value="0FPS",
    label="Response Time",
    sublabel="DESCONOCIDO",
    text_color="#002b5b",
    icon_path="assets/time_icon.png"
)
shadow_s3.grid(row=0, column=2, padx=40)
stat3.grid(row=0, column=2, padx=40)

# === CONFIGURACIÓN DE CÁMARA ===
# Índice de cámara por defecto:
# 0 = cámara integrada (Mac)
# 1, 2, etc. = cámaras externas (iPhone, USB, etc.)
DEFAULT_CAMERA_INDEX = 0  # 👈 fuerza el uso de la cámara interna
USE_DEFAULT_CAMERA = True  # Cambia a False si quieres usar seleccionar_camara()

# === INICIAR CÁMARA ===
if USE_DEFAULT_CAMERA:
    indice = DEFAULT_CAMERA_INDEX
    print(f"🎥 Usando cámara por defecto (índice {indice})")
else:
    indice = seleccionar_camara()
cap = cv2.VideoCapture(indice)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

# === FUNCIÓN DE ACTUALIZACIÓN DE VIDEO ===
def update_frames():
    ret, frame = cap.read()
    if not ret:
        app.after(10, update_frames)
        return

    start_time = time.time()

    # --- YOLO + procesamiento ---
    results = modelo_yolo(frame)[0]
    output, resultado = procesar_deteccion(frame, results, interpreter, input_details, output_details)
    
    # === ACTUALIZAR ESTADÍSTICAS ===
    if resultado:
        etiqueta, nivel_llenado, confianza = resultado

        # --- Fill Level ---
        stat1_value_label.configure(text=f"{nivel_llenado:.1f}%")
        stat1_sublabel.configure(text=etiqueta.upper())

        # --- Confidence ---
        if confianza >= 85:
            confianza_label = "HIGH"
        elif confianza >= 60:
            confianza_label = "MEDIUM"
        else:
            confianza_label = "LOW"

        stat2_value_label.configure(text=f"{confianza:.1f}%")
        stat2_sublabel.configure(text=confianza_label)

    # --- Mostrar cámara original ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(480, 360))
    camera_label.configure(image=img_tk)
    camera_label.image = img_tk

    # --- Mostrar vista procesada ---
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    out_pil = Image.fromarray(output_rgb)
    out_tk = ctk.CTkImage(light_image=out_pil, dark_image=out_pil, size=(480, 360))
    ai_label.configure(image=out_tk)
    ai_label.image = out_tk

    # --- Calcular FPS (opcional) ---
    fps = 1.0 / (time.time() - start_time)
    stat3_value_label.configure(text=f"{int(fps)} FPS")

    # Categoría de velocidad
    if fps > 30:
        stat3_sublabel.configure(text="FAST")
    elif fps > 15:
        stat3_sublabel.configure(text="NORMAL")
    else:
        stat3_sublabel.configure(text="SLOW")
    # stat3.configure(text=f"{int(fps)} FPS")  # actualiza la tarjeta de rendimiento

    # --- Actualizar cada 30 ms ---
    app.after(30, update_frames)

# === INICIAR LOOP DE VIDEO ===
update_frames()
app.mainloop()

# === LIBERAR RECURSOS ===
cap.release()
cv2.destroyAllWindows()