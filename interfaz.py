import customtkinter as ctk
from PIL import Image

# ---------- CONFIGURACIÓN GENERAL ----------
ctk.set_appearance_mode("light") 
ctk.set_default_color_theme("blue")  

app = ctk.CTk()
app.title("AquaSight - Level Sense AI Detector")
app.geometry("1300x800")
app.resizable(False, False)
app.configure(fg_color="#FAFAFA")

# ---------- TOPBAR ----------
from PIL import ImageTk

topbar = ctk.CTkFrame(app, height=60, fg_color="white", corner_radius=0)
topbar.pack(fill="x")

# Logo (a la izquierda)
logo_image = Image.open("assets/icon_logo.png")
logo_image = logo_image.resize((40, 40))
ctk_logo = ctk.CTkImage(light_image=logo_image, dark_image=logo_image, size=(40, 40))

logo_label = ctk.CTkLabel(topbar, image=ctk_logo, text="")
logo_label.pack(side="left", padx=(20, 10), pady=10)

# Nombre de la app
app_name_label = ctk.CTkLabel(
    topbar,
    text="AquaSight",
    font=("Lato", 20, "bold"),
    text_color="#002b5b"
)
app_name_label.pack(side="left", pady=10)

# Versión (a la derecha)
version_label = ctk.CTkLabel(
    topbar,
    text="v3.0.1",
    font=("Lato", 14),
    text_color="#7a7a7a"
)
version_label.pack(side="right", padx=20, pady=10)

# Línea divisoria inferior opcional
divider = ctk.CTkFrame(app, height=1, fg_color="#E0E0E0")
divider.pack(fill="x")

# ---------- HEADER SUPERIOR ----------
header_frame = ctk.CTkFrame(app, fg_color="transparent")
header_frame.pack(pady=40)

title_label = ctk.CTkLabel(
    header_frame,
    text="AquaSight - Level Sense AI Detector",
    font=("Lato", 28, "bold")
)
title_label.pack()

subtitle_label = ctk.CTkLabel(
    header_frame,
    text="AI Vision system for automated filling level control",
    font=("Lato", 16),
    text_color="#4A4A4A"
)
subtitle_label.pack(pady=(10, 20))

# ---------- CONTENEDOR PRINCIPAL ----------
main_frame = ctk.CTkFrame(app, fg_color="transparent")
main_frame.pack(pady=40)

# Asegurar columnas uniformes
main_frame.grid_columnconfigure((0, 1), weight=1, uniform="cards")

# Función para crear una tarjeta
def create_card(parent, icon, title, desc, btn_text, color, hover_color):
    image = Image.open(icon)
    image = image.resize((64, 64))
    ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(64, 64))

    
    # --- Sombra detrás ---
    shadow = ctk.CTkFrame(
        parent,
        width=320,
        height=260,
        corner_radius=20,
        fg_color="#F3F3F3" 
    )
    shadow.grid_propagate(False)
    shadow.pack_propagate(False)

    # --- Tarjeta principal ---
    card = ctk.CTkFrame(
        parent,
        width=320,
        height=260,
        corner_radius=20,
        fg_color="white"
    )
    card.grid_propagate(False)
    card.pack_propagate(False)

    # --- Colocar con pequeño desplazamiento ---
    shadow.place(x=8, y=8)
    card.place(x=0, y=0)

    # --- Contenido interno ---
    image_label = ctk.CTkLabel(card, image=ctk_image, text="")
    image_label.pack(pady=(25, 10))

    title_label = ctk.CTkLabel(card, text=title, font=("Lato", 18, "bold"), text_color="#002b5b")
    title_label.pack()

    desc_label = ctk.CTkLabel(
        card,
        text=desc,
        font=("Lato", 14),
        wraplength=260,
        justify="center",
        text_color="#4A4A4A"
    )
    desc_label.pack(pady=(10, 15))

    button = ctk.CTkButton(
        card,
        text=btn_text,
        corner_radius=8,
        fg_color=color,
        hover_color=hover_color,
        text_color="white",
        width=200,
        height=40,
        font=("Lato", 14, "normal")
    )
    button.pack(pady=(0, 10))

    return shadow, card

# Crear tarjetas
shadow1, realtime_card = create_card(
    main_frame,
    "assets/realtime_icon.png",
    "Real-Time Mode",
    "Live analysis with camera feed for continuous monitoring.",
    "Start Live Detection",
    "#0B2940",
    "#113C5E"
)
shadow1.grid(row=0, column=0, padx=50, pady=10)
realtime_card.grid(row=0, column=0, padx=50, pady=10)

shadow2, image_card = create_card(
    main_frame,
    "assets/image_processing_icon.png",
    "Image Processing",
    "Upload an image file for batch detection and analysis.",
    "Upload Image",
    "#0BADAC",
    "#0DCCCA"
)

shadow2.grid(row=0, column=1, padx=50)
image_card.grid(row=0, column=1, padx=50)


# ---------- SECCIÓN DE ESTADÍSTICAS INFERIOR ----------
stats_frame = ctk.CTkFrame(app, fg_color="transparent")
stats_frame.pack(pady=40)

def create_stat_card(parent, value, label, text_color):
    # --- Sombra detrás ---
    shadow = ctk.CTkFrame(
        parent,
        width=260,
        height=120,
        corner_radius=20,
        fg_color="#F3F3F3"
    )
    shadow.grid_propagate(False)
    shadow.pack_propagate(False)

    # --- Tarjeta principal ---
    card = ctk.CTkFrame(
        parent,
        width=260,
        height=120,
        corner_radius=20,
        fg_color="white"
    )
    card.grid_propagate(False)
    card.pack_propagate(False)

    # --- Colocar con pequeño desplazamiento ---
    shadow.place(x=8, y=8)
    card.place(x=0, y=0)

    # --- Contenido interno ---
    value_label = ctk.CTkLabel(card, text=value, font=("Lato", 24, "bold"), text_color=text_color)
    value_label.pack(pady=(30, 0))

    label_label = ctk.CTkLabel(card, text=label, font=("Lato", 14), text_color="#4A4A4A")
    label_label.pack(pady=(5, 0))

    return shadow, card

# Crear tarjetas de estadísticas
shadow_s1, stat1 = create_stat_card(stats_frame, "98.6%", "Model Accuracy", "#0BADAC")
shadow_s1.grid(row=0, column=0, padx=40)
stat1.grid(row=0, column=0, padx=40)

shadow_s2, stat2 = create_stat_card(stats_frame, "1,247", "Images Processed", "#F76624")
shadow_s2.grid(row=0, column=1, padx=40)
stat2.grid(row=0, column=1, padx=40)

shadow_s3, stat3 = create_stat_card(stats_frame, "45 ms", "Avg Response Time", "#002b5b")
shadow_s3.grid(row=0, column=2, padx=40)
stat3.grid(row=0, column=2, padx=40)

# ---------- FOOTER OPCIONAL ----------
footer_label = ctk.CTkLabel(
    app,
    text="v3.0.1",
    font=("Lato", 12),
    text_color="#7a7a7a"
)
footer_label.pack(side="bottom", pady=10)

# ---------- MAIN LOOP ----------
app.mainloop()