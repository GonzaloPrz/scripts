import os
import re

# Carpeta con los wav y TextGrid
carpeta = "/Users/gp/data/ad_mci_hc/Audios/wavs"

for archivo in os.listdir(carpeta):
    if archivo.endswith(".TextGrid"):
        ruta_textgrid = os.path.join(carpeta, archivo)
        nombre_base = os.path.splitext(archivo)[0]
        ruta_lab = os.path.join(carpeta, nombre_base + ".lab")

        # Leer el TextGrid y extraer el texto de todos los intervalos
        textos = []
        with open(ruta_textgrid, "r", encoding="utf-8") as f:
            contenido = f.read()

            # Buscar todas las líneas que contienen text = "..."
            matches = re.findall(r'text\s*=\s*"(.*?)"', contenido)

            for texto in matches:
                # Ignorar intervalos vacíos
                if texto.strip():
                    textos.append(texto.strip())

        # Combinar todos los textos y guardar en el .lab
        texto_final = " ".join(textos)

        with open(ruta_lab, "w", encoding="utf-8") as f:
            f.write(texto_final)

        print(f"✅ {archivo} → {nombre_base}.lab creado.")

print("🎉 Conversión completa.")