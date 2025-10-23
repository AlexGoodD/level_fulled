import customtkinter as ctk
from modules.ui.topbar import create_topbar
from modules.ui.header import create_header
from modules.ui.cards import create_card
from modules.ui.stats import create_stat_card
import subprocess
import sys

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("AquaSight - Level Sense AI Detector")
app.geometry("1300x800")
app.configure(fg_color="#FAFAFA")

def open_realtime_interface():
    app.destroy()
    subprocess.Popen([sys.executable, "realtime_interface.py"])
    
def open_image_processing_interface():
    app.destroy()
    subprocess.Popen([sys.executable, "image_interface.py"])

# --- TOPBAR ---
create_topbar(app)

# --- HEADER ---
create_header(
    app,
    title="AquaSight - Level Sense AI Detector",
    subtitle="AI Vision system for automated filling level control"
)

# ---------- CONTENEDOR PRINCIPAL ----------
main_frame = ctk.CTkFrame(app, fg_color="transparent")
main_frame.pack(pady=40)

# Asegurar columnas uniformes
main_frame.grid_columnconfigure((0, 1), weight=1, uniform="cards")

# --- TARJETA 1: Real-Time Mode ---
shadow1, realtime_card = create_card(
    main_frame,
    icon="assets/realtime_icon.png",
    title="Real-Time Mode",
    desc="Live analysis with camera feed for continuous monitoring.",
    btn_text="Start Live Detection",
    color="#0B2940",
    hover_color="#113C5E",
    command=open_realtime_interface
)
shadow1.grid(row=0, column=0, padx=50, pady=10)
realtime_card.grid(row=0, column=0, padx=50, pady=10)

# --- TARJETA 2: Image Processing ---
shadow2, image_card = create_card(
    main_frame,
    icon="assets/image_processing_icon.png",
    title="Image Processing",
    desc="Upload an image file for batch detection and analysis.",
    btn_text="Upload Image",
    color="#0BADAC",
    hover_color="#0DCCCA",
    command=open_image_processing_interface
)
shadow2.grid(row=0, column=1, padx=50, pady=10)
image_card.grid(row=0, column=1, padx=50, pady=10)

# ---------- SECCIÓN DE ESTADÍSTICAS ----------
stats_frame = ctk.CTkFrame(app, fg_color="transparent")
stats_frame.pack(pady=40)

# Asegurar columnas uniformes
stats_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="stats")

# --- TARJETA DE ESTADÍSTICA 1 ---
shadow_s1, stat1 = create_stat_card(
    stats_frame,
    value="98.6%",
    label="Model Accuracy",
    text_color="#0BADAC"
)
shadow_s1.grid(row=0, column=0, padx=40)
stat1.grid(row=0, column=0, padx=40)

# --- TARJETA DE ESTADÍSTICA 2 ---
shadow_s2, stat2 = create_stat_card(
    stats_frame,
    value="1,247",
    label="Images Processed",
    text_color="#F76624"
)
shadow_s2.grid(row=0, column=1, padx=40)
stat2.grid(row=0, column=1, padx=40)

# --- TARJETA DE ESTADÍSTICA 3 ---
shadow_s3, stat3 = create_stat_card(
    stats_frame,
    value="45 ms",
    label="Avg Response Time",
    text_color="#002b5b"
)
shadow_s3.grid(row=0, column=2, padx=40)
stat3.grid(row=0, column=2, padx=40)

# --- STATS SECTION ---
stats_frame = ctk.CTkFrame(app, fg_color="transparent")
stats_frame.pack(pady=40)
# shadow, stat = create_stat_card(...)

app.mainloop()