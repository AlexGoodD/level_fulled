import customtkinter as ctk
from PIL import Image
import subprocess
import sys

def create_topbar(parent, logo_path="assets/icon_logo.png", version="v3.0.1", realtime_mode=False, image_processing=False):
    """Crea la barra superior con logo, nombre y versión."""
    
    topbar = ctk.CTkFrame(parent, height=60, fg_color="white", corner_radius=0)
    topbar.pack(fill="x")
    
    # Función
    def go_to_main():
        """Cierra la vista actual y abre la interfaz principal."""
        parent.winfo_toplevel().destroy()
        subprocess.Popen([sys.executable, "main_interface.py"])

    # Logo
    logo_image = Image.open(logo_path)
    logo_image = logo_image.resize((40, 40))
    ctk_logo = ctk.CTkImage(light_image=logo_image, dark_image=logo_image, size=(40, 40))
    
    logo_button = ctk.CTkButton(
        topbar,
        image=ctk_logo,
        text="",
        width=40,
        height=40,
        fg_color="transparent",
        hover_color="#F0F0F0",
        command=go_to_main
    )
    logo_button.pack(side="left", padx=(20, 10), pady=10)

    # Nombre app
    app_name_label = ctk.CTkLabel(
        topbar,
        text="AquaSight",
        font=("Lato", 20, "bold"),
        text_color="#002b5b"
    )
    app_name_label.pack(side="left", pady=10)
    
    # Modo activo
    if realtime_mode or image_processing:
        if realtime_mode:
            mode_text = "Real-Time Mode"
            bg_color = "#E2E8EF"  # simulando #113C5E con 20% opacidad sobre fondo blanco
            text_color = "#0B2940"
        else:
            mode_text = "Image Processing Mode"
            bg_color = "#E0F7F6"  # simulando #0BADAC con 20% opacidad
            text_color = "#0BADAC"

        # Contenedor con color de fondo redondeado (simula badge)
        mode_frame = ctk.CTkFrame(
            topbar,
            fg_color=bg_color,
            corner_radius=15,
            height=30
        )
        mode_frame.pack(side="left", padx=(10, 0), pady=10)

        mode_label = ctk.CTkLabel(
            mode_frame,
            text=mode_text,
            font=("Lato", 13, "bold"),
            text_color=text_color
        )
        mode_label.pack(padx=15, pady=2)

    # Versión
    version_label = ctk.CTkLabel(
        topbar,
        text=version,
        font=("Lato", 14),
        text_color="#7a7a7a"
    )
    version_label.pack(side="right", padx=20, pady=10)

    # Línea divisoria
    divider = ctk.CTkFrame(parent, height=1, fg_color="#E0E0E0")
    divider.pack(fill="x")

    return topbar