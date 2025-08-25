import os, gc, torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor

# -------------------------------
# Configuración de rutas y dispositivo
# -------------------------------
data_dir = Path(Path.home(), 'data', 'ad_mci_hc') if '/Users/gp' in str(Path.home()) else Path('D:', 'CNC_Audio', 'gonza', 'data', 'ad_mci_hc')

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    dtype = torch.float32

model_id = "Qwen/Qwen2.5-7B-Instruct"

# -------------------------------
# Utilidad principal
# -------------------------------
def build_global_summary(descriptions: list, max_new_tokens: int = 512) -> str:
    escenas_texto = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(descriptions)])

    prompt_resumen = (
        "A continuación tienes descripciones parciales de distintas escenas de un mismo video.\n\n"
        f"{escenas_texto}\n\n"
        "Escribe UNA ÚNICA descripción global, fluida y cohesionada del video completo, "
        "en tercera persona, priorizando acciones y objetos; evita redundancias, listados, "
        "marcadores como 'escena', y repeticiones de sintagmas. Usa elipsis y referencias "
        "pronominales cuando sea natural. No inventes datos fuera de lo observable."
    )

    messages = [{"role": "user", "content": prompt_resumen}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    from contextlib import nullcontext
    autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else nullcontext()

    with torch.inference_mode(), autocast_ctx:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            use_cache=True,
        )

    resumen = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    if prompt_resumen in resumen:
        resumen = resumen.split(prompt_resumen)[-1].strip()

    del inputs, output_ids
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return resumen

# -------------------------------
# Modelo
# -------------------------------
print("Cargando modelo y processor...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
    device_map=None
)

model.eval()
model.to(device)
print(f"✓ Modelo cargado en {device} con dtype {dtype}")

fps = 4
model_id = "Qwen/Qwen2.5-VL-72B-Instruct"

# -------------------------------
# Carga de datos
# -------------------------------
video_descriptions = pd.read_csv(data_dir / f'video_descriptions_{fps}_fps_{model_id.split("/")[1].replace(" ","_")}.csv')
video_descriptions_list = video_descriptions['description'].astype(str).tolist()

# -------------------------------
# Generación
# -------------------------------
resumen = build_global_summary(video_descriptions_list, max_new_tokens=512)
print("✓ Resumen global generado.")

pd.DataFrame({"global_summary": [resumen]}).to_csv(data_dir / 'global_video_summary.csv', index=False)
print(f"✓ Guardado en: {data_dir / 'global_video_summary.csv'}")