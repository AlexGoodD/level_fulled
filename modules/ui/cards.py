import customtkinter as ctk
from PIL import Image

def create_card(parent, icon, title, desc, btn_text, color, hover_color, command=None):
    """Crea una tarjeta con ícono, descripción y botón."""
    image = Image.open(icon)
    image = image.resize((64, 64))
    ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(64, 64))

    shadow = ctk.CTkFrame(
        parent,
        width=320,
        height=260,
        corner_radius=20,
        fg_color="#F3F3F3"
    )
    shadow.grid_propagate(False)
    shadow.pack_propagate(False)
    
    card = ctk.CTkFrame(
         parent, width=320,
         height=260,
         corner_radius=20,
         fg_color="white"
        )
    card.grid_propagate(False)
    card.pack_propagate(False)

    shadow.place(x=8, y=8)
    card.place(x=0, y=0)

    image_label = ctk.CTkLabel(card, image=ctk_image, text="")
    image_label.pack(pady=(25, 10))

    title_label = ctk.CTkLabel(card, text=title, font=("Lato", 18, "bold"), text_color="#002b5b")
    title_label.pack()

    desc_label = ctk.CTkLabel(card, text=desc, font=("Lato", 14), wraplength=260, justify="center", text_color="#4A4A4A")
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
        font=("Lato", 14, "normal"),
        command=command
    )
    button.pack(pady=(0, 10))

    return shadow, card

def create_card_view(parent, icon, title, bar_color="#0BADAC"):
    """Crea una tarjeta con barra superior redondeada arriba y plana abajo, compatible con .grid()."""

    # --- Sombra detrás ---
    shadow = ctk.CTkFrame(
        parent,
        width=500,
        height=430,
        corner_radius=20,
        fg_color="#F3F3F3"
    )
    shadow.grid_propagate(False)
    shadow.pack_propagate(False)

    # --- Tarjeta principal ---
    card = ctk.CTkFrame(
        parent,
        width=500,
        height=430,
        corner_radius=20,
        fg_color="white"
    )
    card.grid_propagate(False)
    card.pack_propagate(False)

    # --- Barra superior de color ---
    top_bar = ctk.CTkFrame(
        card,
        height=50,
        corner_radius=20,
        fg_color=bar_color
    )
    top_bar.pack(fill="x", side="top")

    # --- El truco: ocultar las esquinas inferiores del top_bar ---
    # Esto hace que se vea redondeado arriba y plano abajo 👇
    try:
        top_bar._canvas.configure(highlightthickness=0)
        top_bar._canvas.itemconfig(top_bar._corner_items["bottom_left"], state="hidden")
        top_bar._canvas.itemconfig(top_bar._corner_items["bottom_right"], state="hidden")
    except Exception:
        pass  # Evita errores si la implementación interna cambia

    # --- Icono + texto ---
    image = Image.open(icon)
    image = image.resize((35, 35))  # 🔹 Icono más grande
    ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(35, 35))

    top_container = ctk.CTkFrame(top_bar, fg_color="transparent")
    top_container.pack(side="left", padx=15, pady=8)

    icon_label = ctk.CTkLabel(top_container, image=ctk_image, text="")
    icon_label.pack(side="left", padx=(0, 8))

    title_label = ctk.CTkLabel(
        top_container,
        text=title,
        font=("Lato", 16, "bold"),
        text_color="white"
    )
    title_label.pack(side="left")

    # --- Contenedor de contenido dinámico ---
    content_frame = ctk.CTkFrame(
        card,
        fg_color="#D9D9D9",
        corner_radius=12,
        width=280,
        height=180
    )
    content_frame.pack(padx=20, pady=20, fill="both", expand=True)

    return shadow, card, content_frame