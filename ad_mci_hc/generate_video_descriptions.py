import torch
import cv2
from transformers import AutoModel, AutoProcessor
from PIL import Image
import os

# --- Configuration ---
# Use a GPU if available for much faster inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen2.5-VL-7B-Instruct" # Example model ID from Hugging Face

# --- Load the Model and Processor ---
# The processor handles all the pre-processing of text and images to match what the model expects.
print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True
).eval()
print("✓ Model and processor loaded.")


# --- Part 1: Video Frame Extraction ---
def extract_frames_from_video(video_path: str, fps: int = 1) -> list:
    """
    Extracts frames from a video file at a specified frame rate.
    
    Args:
        video_path: Path to the local video file.
        fps: Frames per second to extract.
        
    Returns:
        A list of PIL Image objects.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")
        
    print(f"Extracting frames from '{video_path}' at {fps} FPS...")
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            # Convert frame from BGR (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb_frame))
        frame_count += 1
        
    cap.release()
    print(f"✓ Extracted {len(frames)} frames.")
    return frames

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
                {"type": "image", "image": img} for img in frames
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
        res = model.generate(**inputs, max_new_tokens=1024)
    
    response = processor.decode(res[0], skip_special_tokens=True)
    
    # The output might include the prompt, so we clean it up
    # This part may need adjustment based on the exact model output format
    clean_response = response.split(prompt)[-1].strip()
    
    return clean_response

# --- Main Execution ---
if __name__ == "__main__":
    # NOTE: You need to have a video file locally for this to work.
    # For example, download a short video and save it as "my_video.mp4"

    local_video_path = "fugu_video.mp4" 

    try:
        # 1. Extract frames from your local video file
        video_frames = extract_frames_from_video(local_video_path, fps=1)
        
        # 2. Define your editable prompt
        editable_prompt = "Por favor, haz una descripción objetiva y tan precisa del contenido de este video."
        
        # 3. Generate the description
        description = describe_video_with_qwen(video_frames, editable_prompt)
        
        print("\n--- Generated Video Description ---")
        print(description)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please download a video file and place it in the same directory as this script, or update the 'local_video_path'.")