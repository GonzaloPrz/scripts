# --- mueve esto ARRIBA DEL TODO, antes de importar torch/transformers ---
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
# -----------------------------------------------------------------------

import torch, gc
import cv2
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import pandas as pd
from pathlib import Path
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

device = "cpu"
fps = 4
model_id = "Qwen/Qwen2.5-VL-72B-Instruct"

print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,cache_dir='D:\CNC_Audio\gonza')

# ↓ Usa SDPA (rápido) y bf16; el modelo se carga UNA sola vez y ya queda en cuda:0
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa", 
    cache_dir='D:\CNC_Audio\gonza'  # <- sin flash-attn
)
model.eval()
print("✓ Model and processor loaded.")

# --- helper: muestreo uniforme de frames y downscale ---
def uniform_sample(lst, k):
    if len(lst) <= k:
        return lst
    step = len(lst) / k
    return [lst[int(i*step)] for i in range(k)]

# Ajusta tamaño de entrada (reduce coste)
# Opcional: si quieres forzar un tamaño menor:
processor.image_processor.size = {"shortest_edge": 448}  # o 384 si aún muy lento

def extract_frames_by_scene(video_path: str, threshold: float = 30.0, fps: int = 1):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frames_by_scene = []

    for i, (start, end) in enumerate(scene_list):
        frames = []
        start_frame = int(start.get_frames())
        end_frame = int(end.get_frames())
        # si fps_video es alto, baja la frecuencia real de muestreo
        frame_interval = max(1, int(fps_video // fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
       
        frames_by_scene.append(frames)
        print(f"Escena {i+1}: {len(frames)} frames usados.")

    cap.release()
    return frames_by_scene

def describe_video_with_qwen(frames: list, prompt: str, max_new_tokens: int = 512) -> str:
    # construye el mensaje
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": f} for f in frames] + [{"type": "text", "text": prompt}],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # preprocesa y sube a GPU
    inputs = processor(text=[text], images=frames, return_tensors="pt").to(device)

    # generación rápida (greedy, cache on) y en bf16
    with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,      # greedy = más rápido/estable; sube si quieres algo más creativo
            do_sample=False,
            use_cache=True,       # acelera el decode
        )

    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    clean = response.split(prompt)[-1].strip() if prompt in response else response.strip()

    # limpia memoria temporal
    del inputs, output_ids
    gc.collect()
    torch.cuda.empty_cache()

    return clean

if __name__ == "__main__":
    data_dir = Path(Path.home(), 'data', 'ad_mci_hc') if '/Users/gp' in str(Path.home()) else Path('D:', 'CNC_Audio', 'gonza', 'data', 'ad_mci_hc')
    if Path(Path(data_dir, f'video_descriptions_{fps}_fps_{model_id.split("/")[1].replace(" ","_")}.csv')).exists():
        descriptions_df = pd.read_csv(Path(data_dir,f'video_descriptions_{fps}_fps_{model_id.split("/")[1].replace(" ","_")}.csv' ))
        descriptions = descriptions_df["description"].to_list()
    else:
        descriptions = []

    local_video_path = "fugu_video.mp4"

    video_frames = extract_frames_by_scene(local_video_path, fps=fps)
    
    editable_prompt = "Por favor, haz una descripción objetiva y tan precisa como puedas del contenido de este video. Evita frases como 'El video muestra' o 'Una secuencia de imágenes' o 'La escena muestra' o 'En esta secuencia' y explicalo de forma secuencial, sin caer en descripciones de imágenes estáticas, sino prestando atención a la dinámica del video. Pon énfasis en las acciones del video y en los objetos que aparecen en él. Por ejemplo, si hay un pez globo, menciona que es un pez globo y describe su comportamiento. Si hay una persona cocinando, describe qué está cocinando y cómo lo hace. No incluyas información sobre el contexto del video, como quién lo grabó o dónde fue grabado. Concéntrate únicamente en lo que se ve en el video."

    for i, frames in enumerate(video_frames[1:-1]):
        if len(frames) > 19:
            frames = uniform_sample(frames, 19)
        if i < len(descriptions):
            print(f"Skipping scene {i + 1}")
            continue

        print(f"Procesando escena con {len(frames)} frames...")
        desc = describe_video_with_qwen(
            frames,
            editable_prompt + "Evita repeticiones de personajes que ya fueron mencionados en escenas previas y usa las descripciones previas para darle cohesión a la descripción completa del video, evitando las repeticiones de frases (sintagmas nominales) y empleando, en cambio, elipsis, sinonimia, referencia mediante pronombres, etcétera.",
            max_new_tokens=384,   # prueba 384; si aún lento, 256
        )
        descriptions.append(desc)
        descriptions_df = pd.DataFrame(descriptions, columns=["description"])

        descriptions_df.to_csv(Path(data_dir, f'video_descriptions_{fps}_fps_{model_id.split("/")[1].replace(" ","_")}.csv'), index=False)