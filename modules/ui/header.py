import customtkinter as ctk

def create_header(parent, title, subtitle):
    """
    Crea un encabezado (header) con título y subtítulo centrados.
    
    Args:
        parent: widget padre (normalmente la ventana principal o un frame)
        title: texto principal grande
        subtitle: texto descriptivo debajo del título
    """
    header_frame = ctk.CTkFrame(parent, fg_color="transparent")
    header_frame.pack(pady=40)

    title_label = ctk.CTkLabel(
        header_frame,
        text=title,
        font=("Lato", 28, "bold"),
        text_color="#002b5b"
    )
    title_label.pack()

    subtitle_label = ctk.CTkLabel(
        header_frame,
        text=subtitle,
        font=("Lato", 16),
        text_color="#4A4A4A"
    )
    subtitle_label.pack(pady=(10, 20))

    return header_frame