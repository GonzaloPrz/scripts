import torch, gc, os
from transformers import AutoModel, AutoProcessor
import pandas as pd
from pathlib import Path

def build_global_summary(descriptions: list, max_new_tokens: int = 512) -> str:
    """
    Toma las descripciones de cada escena y genera una sola descripción continua
    y cohesionada del video completo (sin listas, sin "escena X", sin repeticiones).
    """
    # armamos un texto numerado breve para dar contexto al modelo
    escenas_texto = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(descriptions)])

    prompt_resumen = (
        "A continuación tienes descripciones parciales de distintas escenas de un mismo video.\n\n"
        f"{escenas_texto}\n\n"
        "Escribe UNA ÚNICA descripción global, fluida y cohesionada del video completo, "
        "en tercera persona, priorizando acciones y objetos; evita redundancias, listados, "
        "marcadores como 'escena', y repeticiones de sintagmas. Usa elipsis y referencias "
        "pronominales cuando sea natural. No inventes datos fuera de lo observable."
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_resumen}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # texto puro (sin imágenes). Pasamos images=[] para mantener la misma interfaz.
    inputs = processor(text=[text], images=[], return_tensors="pt").to(device)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            use_cache=True,
        )

    resumen = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    resumen = resumen.split(prompt_resumen)[-1].strip()

    del inputs, output_ids
    gc.collect()
    torch.cuda.empty_cache()
    return resumen

data_dir = Path(Path.home(), 'data', 'ad_mci_hc') if '/Users/gp' in str(Path.home()) else Path('D:', 'CNC_Audio', 'gonza', 'data', 'ad_mci_hc')

video_descriptions = pd.read_csv(data_dir / 'video_descriptions.csv')

video_descriptions_list = video_descriptions['description'].tolist()

torch.cuda.set_device(0)
device = "cuda:0"

model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"

print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# ↓ Usa SDPA (rápido) y bf16; el modelo se carga UNA sola vez y ya queda en cuda:0
model = AutoModel.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",   # <- sin flash-attn
)
model.eval()

model.to(device)
print("✓ Model and processor loaded.")

resumen = build_global_summary(video_descriptions_list, max_new_tokens=512)
print("Global summary generated.")

resumen.to_csv(data_dir / 'global_video_summary.csv', index=False)

