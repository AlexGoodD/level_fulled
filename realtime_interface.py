import customtkinter as ctk
from modules.ui.topbar import create_topbar
from modules.ui.header import create_header
from modules.ui.cards import create_card, create_card_view
from modules.ui.stats import create_stat_card, create_stat_card_view

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

# --- TARJETA 1: Real-Time Mode ---
shadow_view_1, real_view, content = create_card_view(
    main_frame,
    icon="assets/camera_icon.png",
    title="Camera View",
    bar_color="#0BADAC"
)
shadow_view_1.grid(row=0, column=0, padx=50, pady=0)
real_view.grid(row=0, column=0, padx=50, pady=0)


# --- TARJETA 2: Image Processing ---
shadow_view_2, ai_view, content = create_card_view(
    main_frame,
    icon="assets/ai_icon.png",
    title="AI View",
    bar_color="#F76624"
)
shadow_view_2.grid(row=0, column=1, padx=50, pady=0)
ai_view.grid(row=0, column=1, padx=50, pady=0)

# ---------- SECCIÓN DE ESTADÍSTICAS ----------
stats_frame = ctk.CTkFrame(app, fg_color="transparent")
stats_frame.pack(pady=40)

# Asegurar columnas uniformes
stats_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="stats")

# --- TARJETA DE ESTADÍSTICA 1 ---
shadow_s1, stat1 = create_stat_card_view(
    stats_frame,
    value="98.6%",
    label="Model Accuracy",
    text_color="#0BADAC"
)
shadow_s1.grid(row=0, column=0, padx=40)
stat1.grid(row=0, column=0, padx=40)

# --- TARJETA DE ESTADÍSTICA 2 ---
shadow_s2, stat2 = create_stat_card_view(
    stats_frame,
    value="1,247",
    label="Images Processed",
    text_color="#F76624"
)
shadow_s2.grid(row=0, column=1, padx=40)
stat2.grid(row=0, column=1, padx=40)

# --- TARJETA DE ESTADÍSTICA 3 ---
shadow_s3, stat3 = create_stat_card_view(
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