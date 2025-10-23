import cv2
import time
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog
import numpy as np

# === MODULOS DE CAMARAS / MODELOS ===
from modules.models import cargar_modelo_clasificacion, cargar_modelo_yolo
from modules.detector import procesar_deteccion

# === MODULOS DE UI ===
from modules.ui.topbar import create_topbar
from modules.ui.cards import create_card_view
from modules.ui.stats import create_stat_card_view
from modules.ui.charts import create_chart_card, draw_accuracy, draw_confusion
from matplotlib import pyplot as plt

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
create_topbar(app, image_processing=True)

# ---------- CONTENEDOR PRINCIPAL ----------
main_frame = ctk.CTkFrame(app, fg_color="transparent")
main_frame.pack(pady=40)

# Asegurar columnas uniformes
main_frame.grid_columnconfigure((0, 1), weight=1, uniform="cards")

# --- TARJETA 1: Selector de imagen (antes: cámara) ---
shadow_view_1, real_view, camera_content = create_card_view(
    main_frame,
    icon="assets/camera_icon.png",
    title="Image View",         
    bar_color="#0BADAC"
)
shadow_view_1.grid(row=0, column=0, padx=50, pady=0)
real_view.grid(row=0, column=0, padx=50, pady=0)

# --- ETIQUETAS PARA MOSTRAR IMAGENES ---
camera_label = ctk.CTkLabel(
    camera_content,
    text="Haz clic para seleccionar una imagen",
    width=480,
    height=360,
    fg_color="#FFFFFF",
    corner_radius=12,
    font=("Lato", 14),
    text_color="#4A4A4A",
    bg_color="#FFFFFF",
)
camera_label.pack(expand=True, fill="both")

# ================= MODIFICACIÓN SOLICITADA =================
# Contenedor en la celda (row=0, column=1) para apilar arriba/abajo
right_col = ctk.CTkFrame(main_frame, fg_color="transparent")
right_col.grid(row=0, column=1, padx=30, pady=0, sticky="nsew")

# Permitir que el contenedor use el espacio vertical disponible
main_frame.grid_columnconfigure(1, weight=1)
main_frame.grid_rowconfigure(0, weight=1)
right_col.grid_rowconfigure(0, weight=1)
right_col.grid_rowconfigure(1, weight=1)
right_col.grid_columnconfigure(0, weight=1)

# --- TARJETA: Accuracy vs Epochs (arriba) ---
acc_shadow, acc_card, acc_content, acc_fig, acc_ax, acc_canvas = create_chart_card(
    right_col, icon_path="assets/chart_icon.png", title="Accuracy vs Epochs",
    bar_color="transparent", width=500, height=200
)
acc_shadow.grid(row=0, column=0, padx=0, pady=(0, 6), sticky="n")
acc_card.grid(row=0, column=0, padx=0, pady=(0, 6), sticky="n")

# --- TARJETA: Confusion Matrix (abajo) ---
cm_shadow, cm_card, cm_content, cm_fig, cm_ax, cm_canvas = create_chart_card(
    right_col, icon_path="assets/matrix_icon.png", title="Confusion Matrix",
    bar_color="transparent", width=500, height=200
)
cm_shadow.grid(row=1, column=0, padx=0, pady=(6, 0), sticky="s")
cm_card.grid(row=1, column=0, padx=0, pady=(6, 0), sticky="s")
# ================= FIN MODIFICACIÓN =================

# ====== PLACEHOLDERS INICIALES ======
_demo_epochs = list(range(1, 11))
_demo_train_acc = np.linspace(0.5, 0.92, num=10)
_demo_val_acc = np.linspace(0.48, 0.89, num=10)
draw_accuracy(acc_ax, acc_canvas, _demo_epochs, _demo_train_acc, _demo_val_acc)

_demo_labels = ["EMPTY", "HALF", "FULL", "SPILLED"]
_demo_true = np.random.choice(_demo_labels, size=120, p=[0.25, 0.25, 0.4, 0.1])
_demo_pred = np.random.choice(_demo_labels, size=120, p=[0.2, 0.3, 0.4, 0.1])
draw_confusion(cm_ax, cm_canvas, _demo_true, _demo_pred, _demo_labels)

# ---------- SECCIÓN DE ESTADÍSTICAS ----------
stats_frame = ctk.CTkFrame(app, fg_color="transparent")
stats_frame.pack(pady=40)
stats_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="stats")

shadow_s1, stat1, stat1_value_label, stat1_sublabel = create_stat_card_view(
    stats_frame,
    value="0%",
    label="Fill Level",
    sublabel="UNKNOWN",
    text_color="#0BADAC",
    icon_path="assets/fill_icon.png"
)
shadow_s1.grid(row=0, column=0, padx=40)
stat1.grid(row=0, column=0, padx=40)

shadow_s2, stat2, stat2_value_label, stat2_sublabel = create_stat_card_view(
    stats_frame,
    value="0%",
    label="Confidence",
    sublabel="UNKNOWN",
    text_color="#F76624",
    icon_path="assets/check_icon.png"
)
shadow_s2.grid(row=0, column=1, padx=40)
stat2.grid(row=0, column=1, padx=40)

shadow_s3, stat3, stat3_value_label, stat3_sublabel = create_stat_card_view(
    stats_frame,
    value="0FPS",
    label="Response Time",
    sublabel="UNKNOWN",
    text_color="#002b5b",
    icon_path="assets/time_icon.png"
)
shadow_s3.grid(row=0, column=2, padx=40)
stat3.grid(row=0, column=2, padx=40)

# --- Función para actualizar gráficas post inferencia ---
def actualizar_tarjetas_metricas_post_inferencia():
    global _demo_epochs, _demo_train_acc, _demo_val_acc
    # Simular nueva época para la demo
    _demo_epochs.append(_demo_epochs[-1] + 1)
    _demo_train_acc = np.append(_demo_train_acc, min(0.99, _demo_train_acc[-1] + np.random.uniform(-0.01, 0.02)))
    _demo_val_acc = np.append(_demo_val_acc, min(0.99, _demo_val_acc[-1] + np.random.uniform(-0.015, 0.02)))
    draw_accuracy(acc_ax, acc_canvas, _demo_epochs, _demo_train_acc, _demo_val_acc)

# === NUEVO: Función para cargar imagen ===
def cargar_imagen(event=None):
    filepath = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Imagenes", "*.png;*.jpg;*.jpeg;*.bmp"), ("Todos", "*.*")]
    )
    if not filepath:
        return

    try:
        img_pil = Image.open(filepath).convert("RGB")
    except Exception as e:
        print(f"❌ No se pudo abrir la imagen: {e}")
        return

    # Previsualización en la tarjeta
    preview = img_pil.copy()
    preview.thumbnail((480, 360))
    img_tk = ctk.CTkImage(light_image=preview, dark_image=preview, size=(480, 360))
    camera_label.configure(image=img_tk, text="")
    camera_label.image = img_tk  # mantener referencia

    # Ejecutar modelo y actualizar estadísticas
    frame_rgb = np.array(img_pil)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    start_time = time.time()
    results = modelo_yolo(frame_bgr)[0]
    _, resultado = procesar_deteccion(frame_bgr, results, interpreter, input_details, output_details)
    elapsed = time.time() - start_time

    if resultado:
        etiqueta, nivel_llenado, confianza = resultado
        stat1_value_label.configure(text=f"{nivel_llenado:.1f}%")
        stat1_sublabel.configure(text=etiqueta.upper())
        if confianza >= 85:
            confianza_label = "HIGH"
        elif confianza >= 60:
            confianza_label = "MEDIUM"
        else:
            confianza_label = "LOW"
        stat2_value_label.configure(text=f"{confianza:.1f}%")
        stat2_sublabel.configure(text=confianza_label)

    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    stat3_value_label.configure(text=f"{int(fps)} FPS")
    if fps > 30:
        stat3_sublabel.configure(text="FAST")
    elif fps > 15:
        stat3_sublabel.configure(text="NORMAL")
    else:
        stat3_sublabel.configure(text="SLOW")
        
        actualizar_tarjetas_metricas_post_inferencia()

# --- PLACEHOLDER CLICABLE (abre el dialogo) ---
camera_label.bind("<Button-1>", cargar_imagen)


# === INICIAR LOOP DE UI ===
app.mainloop()
