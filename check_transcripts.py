import os

# Ruta a tu carpeta con los wav y TextGrid
carpeta = "/Users/gp/data/ad_mci_hc/Audios/wavs"

errores = []

for archivo in os.listdir(carpeta):
    if archivo.endswith(".TextGrid"):
        ruta = os.path.join(carpeta, archivo)
        wav_correspondiente = os.path.splitext(archivo)[0] + ".wav"

        # 1. Verifica si existe el WAV correspondiente
        if not os.path.exists(os.path.join(carpeta, wav_correspondiente)):
            errores.append(f"❌ Falta el archivo WAV para {archivo}")

        # 2. Verifica si el TextGrid está vacío
        if os.path.getsize(ruta) == 0:
            errores.append(f"❌ El archivo {archivo} está vacío.")

        # 3. Intenta leer el contenido
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                contenido = f.read()
                if "IntervalTier" not in contenido and "TextTier" not in contenido:
                    errores.append(f"⚠ El archivo {archivo} no parece tener tiers válidos.")
        except UnicodeDecodeError:
            errores.append(f"❌ El archivo {archivo} tiene una codificación no UTF-8 o caracteres inválidos.")

# Mostrar resultados
if errores:
    print("🚨 Problemas encontrados:")
    for error in errores:
        print(error)
else:
    print("✅ Todos los TextGrid parecen estar bien formados y coinciden con sus WAV.")