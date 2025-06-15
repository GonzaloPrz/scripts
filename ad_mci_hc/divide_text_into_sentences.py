import stanza
import pandas as pd
from pathlib import Path

# Descargar el modelo de español la primera vez
stanza.download('es')

# Cargar el pipeline en español
nlp = stanza.Pipeline(lang='es', processors='tokenize')

def segmentar_oraciones_stanza(texto):
    doc = nlp(texto)
    oraciones = [sent.text for sent in doc.sentences]
    return oraciones

data_dir = Path(Path.home(),'data','ad_mci_hc') if '/Users/gp' in str(Path.home()) else Path('D','CNC_Audio','gonza','data','ad_mci_hc')

textos = pd.read_csv()
# Ejemplo de uso
texto = """El cocinero pidió un pescado lo llevó para cocinarlo lo puso en la tabla para prepararlo para la cocción 
el pez se asustó tanto que salió volando era un pez globo salió volando volando se fue por el living 
había un cuadro en la muralla que era el mar y él pensó que era el agüita y se tiró contra el cuadro y cayó."""

oraciones = segmentar_oraciones_stanza(texto)

for i, o in enumerate(oraciones, 1):
    print(f"{i}: {o}")
