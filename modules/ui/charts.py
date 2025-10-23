import customtkinter as ctk
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix

def create_chart_card(parent, icon_path, title,
                      bar_color="#E9E9E9",
                      width=420, height=160,
                      fig_w=4.0, fig_h=1.6, fig_dpi=100):
    shadow = ctk.CTkFrame(parent, width=width, height=height, corner_radius=20, fg_color="#F3F3F3")
    shadow.grid_propagate(False); shadow.pack_propagate(False)

    card = ctk.CTkFrame(parent, width=width, height=height, corner_radius=20, fg_color="white")
    card.grid_propagate(False); card.pack_propagate(False)

    top_bar = ctk.CTkFrame(card, height=36, corner_radius=20, fg_color=bar_color)
    top_bar.pack(fill="x", side="top")
    try:
        top_bar._canvas.configure(highlightthickness=0)
        top_bar._canvas.itemconfig(top_bar._corner_items["bottom_left"], state="hidden")
        top_bar._canvas.itemconfig(top_bar._corner_items["bottom_right"], state="hidden")
    except Exception:
        pass

    try:
        icon_img = Image.open(icon_path).resize((20, 20))
        ctk_icon = ctk.CTkImage(light_image=icon_img, dark_image=icon_img, size=(20, 20))
    except Exception:
        ctk_icon = None

    head = ctk.CTkFrame(top_bar, fg_color="transparent")
    head.pack(side="left", padx=10, pady=4)
    if ctk_icon:
        ctk.CTkLabel(head, image=ctk_icon, text="").pack(side="left", padx=(0, 6))
    ctk.CTkLabel(head, text=title, font=("Lato", 13, "bold"), text_color="#0B2940").pack(side="left")

    # Contenedor del gráfico: que se expanda y mantenga altura
    inner_w = max(100, width - 28*2)
    inner_h = max(80, height - 36 - 16*2)
    content = ctk.CTkFrame(card, fg_color="white", corner_radius=12, width=inner_w, height=inner_h)
    content.pack(padx=16, pady=12, fill="both", expand=True)
    content.pack_propagate(False)

    # Figura con layout que evita recortes
    fig = Figure(figsize=(fig_w, fig_h), dpi=fig_dpi, layout="constrained")
    fig.set_constrained_layout(True)  # asegura que textos/leyendas/colorbar no se corten
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    fig.patch.set_alpha(0)

    canvas = FigureCanvasTkAgg(fig, master=content)
    widget = canvas.get_tk_widget()
    widget.pack(fill="both", expand=True)  # ocupar todo el alto disponible
    canvas.draw()

    return shadow, card, content, fig, ax, canvas

def draw_accuracy(ax, canvas, epochs, train_acc, val_acc=None):
    ax.clear()
    ax.set_title("Accuracy vs Epochs", fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.plot(epochs, train_acc, color="#0BADAC", linewidth=2, label="Train")
    if val_acc is not None:
        ax.plot(epochs, val_acc, color="#F76624", linewidth=2, label="Val")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    # Ajuste final para que nada quede fuera del canvas
    try:
        ax.figure.tight_layout()
    except Exception:
        pass
    canvas.draw()

def draw_confusion(ax, canvas, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ax.clear()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)

    # Mantener celdas cuadradas y adaptar a la caja disponible
    ax.set_aspect("equal", adjustable="box")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8)

    # Limpiar colorbar anterior y crear uno nuevo sin cortar el contenido
    if hasattr(ax.figure, "_cm_colorbar"):
        ax.figure._cm_colorbar.remove()
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.figure._cm_colorbar = cb

    try:
        ax.figure.tight_layout()
    except Exception:
        pass
    canvas.draw()