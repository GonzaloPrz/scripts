import torch
import cv2
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import os
import pandas as pd
from pathlib import Path
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from joblib import Parallel, delayed

data_dir = Path(Path.home(), 'data', 'ad_mci_hc') if '/Users/gp' in str(Path.home()) else Path('D:', 'CNC_Audio', 'gonza', 'data', 'ad_mci_hc')
# --- Configuration ---
# Use a GPU if available for much faster inference
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

model_id = "Qwen/Qwen2.5-VL-7B-Instruct" # Example model ID from Hugging Face

# --- Load the Model and Processor ---
# The processor handles all the pre-processing of text and images to match what the model expects.
print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    trust_remote_code=True
).eval()
print("✓ Model and processor loaded.")


# --- Part 1: Video Frame Extraction ---
def extract_frames_by_scene(video_path: str, threshold: float = 30.0, fps: int = 1):
    """
    Extrae los frames de cada escena detectada en el video.
    Devuelve una lista de listas: cada sublista contiene los frames (PIL.Image) de una escena.
    """
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    import cv2
    from PIL import Image

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
        frame_interval = int(fps_video // fps) if fps > 0 else 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb_frame))
        frames_by_scene.append(frames)
        print(f"Escena {i+1}: {len(frames)} frames extraídos.")

    cap.release()
    return frames_by_scene

# --- Part 2: Generate Description with Qwen2.5-VL ---
def describe_video_with_qwen(frames: list, prompt: str) -> str:
    """
    Generates a description for a sequence of video frames using Qwen-VL.
    """
    print("Generating description with Qwen2.5-VL...")
    
    # Format the input for the model
    # The processor expects a list of dictionaries with specific keys
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame} for frame in frames
            ] + [
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt")
    inputs = inputs.to(device)

    # Generate the response
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1024)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    clean_response = response.split(prompt)[-1].strip() if prompt in response else response.strip()
    return clean_response
# --- Main Execution ---
if __name__ == "__main__":
    # NOTE: You need to have a video file locally for this to work.
    # For example, download a short video and save it as "my_video.mp4"

    local_video_path = "fugu_video.mp4"
    
    all_descriptions = []

    # 1. Extract frames from your local video file
    video_frames = extract_frames_by_scene(local_video_path, fps=1)
    
    # 2. Define your editable prompt
    editable_prompt = "Por favor, haz una descripción objetiva y tan precisa como puedas del contenido de este video. Evita frases como 'El video muestra' o 'Una secuencia de imágenes' y explicalo de forma secuencial, sin caer en descripciones de imágenes estáticas, sino prestando atención a la dinámica del video."
    
    # 3. Generate the description
    for frames in video_frames[1:-1]:
        description = describe_video_with_qwen(frames, editable_prompt)
        all_descriptions.append(description)
    all_descriptions = pd.DataFrame(all_descriptions, columns=["description"])

    all_descriptions.to_csv(Path(data_dir,"video_descriptions.csv"), index=False)
    