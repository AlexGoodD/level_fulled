import customtkinter as ctk
from PIL import Image

width = 310
height = 120

def create_stat_card(parent, value, label, text_color):
    shadow = ctk.CTkFrame(
        parent, 
        width=width, 
        height=height, 
        corner_radius=20, 
        fg_color="#F3F3F3"
    )
    shadow.grid_propagate(False)
    shadow.pack_propagate(False)
    
    card = ctk.CTkFrame(
        parent, 
        width=width, 
        height=height, 
        corner_radius=20, 
        fg_color="white"
    )
    card.grid_propagate(False)
    card.pack_propagate(False)

    shadow.place(x=8, y=8)
    card.place(x=0, y=0)

    value_label = ctk.CTkLabel(card, text=value, font=("Lato", 24, "bold"), text_color=text_color)
    value_label.pack(pady=(30, 0))

    label_label = ctk.CTkLabel(card, text=label, font=("Lato", 14), text_color="#4A4A4A")
    label_label.pack(pady=(5, 0))

    return shadow, card

def create_stat_card_view(parent, value, label, sublabel, text_color="#0BADAC", icon_path=None):
    """Crea una tarjeta de estadística con barra lateral, icono y tres textos (valor, etiqueta, subetiqueta)."""

    # --- Sombra detrás ---
    shadow = ctk.CTkFrame(
        parent,
        width=width,
        height=height,
        corner_radius=20,
        fg_color="#F3F3F3"
    )
    shadow.grid_propagate(False)
    shadow.pack_propagate(False)

    # --- Tarjeta principal ---
    card = ctk.CTkFrame(
        parent,
        width=width,
        height=height,
        corner_radius=20,
        fg_color="white"
    )
    card.grid_propagate(False)
    card.pack_propagate(False)

    # --- Posicionar sombra detrás ---
    shadow.place(x=8, y=8)
    card.place(x=0, y=0)

    # --- Barra lateral colorida ---
    color_bar = ctk.CTkFrame(
        card,
        width=6,
        height=120,
        corner_radius=10,
        fg_color=text_color
    )
    color_bar.pack(side="left", fill="y", padx=(0, 10))

    # --- Contenedor principal de texto ---
    text_container = ctk.CTkFrame(card, fg_color="transparent")
    text_container.pack(side="left", padx=(10, 0), pady=15)

    # Valor principal (por ejemplo: 93%)
    value_label = ctk.CTkLabel(
        text_container,
        text=value,
        font=("Lato", 22, "bold"),
        text_color=text_color
    )
    value_label.pack(anchor="w")

    # Etiqueta (por ejemplo: Fill Level)
    label_label = ctk.CTkLabel(
        text_container,
        text=label,
        font=("Lato", 14),
        text_color="#4A4A4A"
    )
    label_label.pack(anchor="w")

    # Subetiqueta (por ejemplo: FULL, HIGH, FAST)
    sublabel_label = ctk.CTkLabel(
        text_container,
        text=sublabel,
        font=("Lato", 14, "bold"),
        text_color=text_color
    )
    sublabel_label.pack(anchor="w")

    # --- Icono a la derecha ---
    if icon_path:
        try:
            image = Image.open(icon_path)
            image = image.resize((40, 40))
            ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(40, 40))
            icon_label = ctk.CTkLabel(card, image=ctk_image, text="")
            icon_label.pack(side="right", padx=10)
        except Exception as e:
            print(f"⚠️ No se pudo cargar el icono: {e}")

    # 🔹 DEVUELVE TODOS LOS ELEMENTOS para poder actualizarlos luego
    return shadow, card, value_label, sublabel_label