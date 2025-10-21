import customtkinter as ctk

def create_stat_card(parent, value, label, text_color):
    shadow = ctk.CTkFrame(
        parent, 
        width=260, 
        height=120, 
        corner_radius=20, 
        fg_color="#F3F3F3"
    )
    shadow.grid_propagate(False)
    shadow.pack_propagate(False)
    
    card = ctk.CTkFrame(
        parent, 
        width=260, 
        height=120, 
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

def create_stat_card_view(parent, value, label, text_color):
    shadow = ctk.CTkFrame(
        parent, 
        width=260, 
        height=120, 
        corner_radius=20, 
        fg_color="#F3F3F3"
    )
    shadow.grid_propagate(False)
    shadow.pack_propagate(False)
    
    card = ctk.CTkFrame(
        parent, 
        width=260, 
        height=120, 
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