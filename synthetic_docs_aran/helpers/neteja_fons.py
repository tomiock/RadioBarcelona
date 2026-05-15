import os
from PIL import Image, ImageOps

def clean_asset_background(input_path, output_path):
    # Obrim la imatge i ens assegurem que estigui en mode RGBA
    img = Image.open(input_path).convert("RGBA")
    
    # Opcional: Millorar el contrast per separar millor el fons de la tinta
    # img = ImageOps.autocontrast(img, cutoff=2)

    data = img.getdata()
    new_data = []

    for item in data:
        # Calculem la brillantor (threshold). 
        # Si el píxel és molt clar (apropat al 255,255,255), el fem transparent.
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            new_data.append((255, 255, 255, 0))  # Transparència total
        else:
            new_data.append(item)

    img.putdata(new_data)
    img.save(output_path, "PNG")

# Exemple d'ús per a la teva carpeta de 'stamps'
folder = "assets/stamps"
for filename in os.listdir(folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        clean_asset_background(f"{folder}/{filename}", f"{folder}/cleaned_{filename}")