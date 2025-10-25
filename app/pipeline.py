# app/pipeline.py
import os
import subprocess
import cv2
import librosa
import numpy as np
from transformers import pipeline
import json
import warnings
import base64
import tempfile

# --- 1. ASR MODEL LOADING ---
# We define a global variable for the pipeline
ASR_PIPELINE = None

def load_model():
    """Loads the ASR model into the global ASR_PIPELINE variable."""
    global ASR_PIPELINE
    try:
        print("[INFO] Loading ASR model (this will run on service startup)...")
        ASR_PIPELINE = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo"
        )
        print("[SUCCESS] ASR model loaded.")
    except Exception as e:
        print(f"[ERROR] Could not load ASR model: {e}")

# ---
# 1. Module A(1): Audio Extraction (via FFmpeg)
# ---
def extract_audio_ffmpeg(video_path: str, audio_output_path: str) -> bool:
    print(f"[INFO] Extracting audio from '{video_path}'...")
    command = [
        'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
        '-ar', '44100', '-ac', '2', '-y', audio_output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[SUCCESS] Audio extracted to '{audio_output_path}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg failed to extract audio: {e}")
        return False
    except FileNotFoundError:
        print("[ERROR] FFmpeg not found.")
        return False

# ---
# 2. Module A(2): Vocal Separation via Demucs 
# ---
def separate_vocals_demucs(audio_path: str, output_dir: str) -> dict:
    """
    Separates vocals from background music/sound using Demucs.
    Uses the 'htdemucs_ft' model (4-stem) and defines 'other.wav' as background.
    """
    print(f"[INFO] Separating vocals using Demucs for '{audio_path}'...")

    model_name = "htdemucs_ft"
    command = [
        'demucs',
        '-n', model_name,   # Model to use
        '-o', output_dir,   # Output directory
        audio_path
    ]

    try:
        # Run demucs
        # NOTE: If it fails again, remove stdout/stderr to see the full error log
        subprocess.run(command, check=True, capture_output=True, text=True)

        # Demucs creates a subfolder structure: [output_dir]/[model_name]/[audio_filename_base]/
        audio_filename_base = os.path.splitext(os.path.basename(audio_path))[0]
        demucs_out_folder = os.path.join(output_dir, model_name, audio_filename_base)

        # Find all generated .wav files
        all_stems = [f for f in os.listdir(demucs_out_folder) if f.endswith('.wav')]
        paths_dict = {
            os.path.splitext(stem)[0]: os.path.join(demucs_out_folder, stem)
            for stem in all_stems
        }

        if 'vocals' in paths_dict:
            print(f"[SUCCESS] Vocals and {len(paths_dict)-1} background stems separated in '{demucs_out_folder}'")
            return paths_dict # Return all paths
        else:
            print(f"[ERROR] Demucs ran, but 'vocals.wav' is missing from '{demucs_out_folder}'.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Demucs separation failed. Command: {e.cmd}")
        print(f"[ERROR] Demucs return code: {e.returncode}")
        print(f"------------------- Demucs stdout -------------------")
        print(e.stdout)
        print(f"------------------- Demucs stderr (THE REAL ERROR) -------------------")
        print(e.stderr) 
        print(f"-----------------------------------------------------------------")
        return None
    except FileNotFoundError:
        print("[ERROR] Demucs not found. Please ensure it is installed (`pip install demucs`).")
        return None


# ---
# 3. Module B: Speech-to-Text (ASR)
# ---
def transcribe_vocals(vocal_audio_path: str) -> str:
    if ASR_PIPELINE is None:
        print("[ERROR] ASR pipeline not available. Skipping transcription.")
        return ""

    print(f"[INFO] Transcribing audio from '{vocal_audio_path}'...")
    try:
        transcription_result = ASR_PIPELINE(vocal_audio_path)
        transcribed_text = transcription_result['text']
        print("[SUCCESS] Transcription complete.")
        return transcribed_text.strip()
    except Exception as e:
        print(f"[ERROR] Error during transcription: {e}")
        return ""

# ---
# 4. Module C: Frame Extraction (OpenCV)
# ---
def extract_video_frames(video_path: str, interval_sec: int = 2) -> list:
    """
    Extracts frames and returns a list of base64 encoded strings.
    The 'output_dir' parameter is no longer used for saving but is kept
    for function signature compatibility.
    """
    print(f"[INFO] Extracting & encoding frames from '{video_path}' at {interval_sec}s intervals...")

    # This list will hold the base64 strings
    base64_frames = []

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return base64_frames

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("[ERROR] Video FPS is 0. Cannot extract frames.")
        return base64_frames

    frame_interval = int(fps * interval_sec)
    success, image = vidcap.read()
    count = 0
    extracted_count = 0

    while success:
        if count % frame_interval == 0:
            # Encode frame as JPEG in-memory
            success_encode, buffer = cv2.imencode('.jpg', image)

            if success_encode:
                # Convert the in-memory buffer to a base64 string
                b64_string = base64.b64encode(buffer).decode('utf-8')
                base64_frames.append(b64_string)
                extracted_count += 1
            else:
                print(f"[WARNING] Failed to encode frame {extracted_count}")

        success, image = vidcap.read()
        count += 1

    vidcap.release()
    print(f"[SUCCESS] Extracted and encoded {extracted_count} frames to base64.")
    return base64_frames

# ---
# 5b. Supporting Module: Merge Background Tracks
# ---

def merge_background_tracks(separation_paths: dict, output_path: str) -> str:
    """
    Merges all non-vocal tracks from Demucs into a single background track
    using ffmpeg's 'amix' filter (mixing tracks together).

    Args:
        separation_paths: Dict from Demucs (e.g., {'vocals': '...', 'bass': '...'}).
        output_path: The file path to save the merged .wav file.

    Returns:
        The output_path if successful, None otherwise.
    """
    # 1. Get all file paths *except* 'vocals'
    background_files = []
    for key, path in separation_paths.items():
        # Add any track that is not 'vocals' and has a valid path
        if key != 'vocals' and path and os.path.exists(path):
            background_files.append(path)

    # 2. Handle edge cases
    if not background_files:
        print("[WARNING] No background tracks found to merge.")
        return None

    # If only one background track exists (e.g., from a 2-stem model),
    # just return that track's path. No merge needed.
    if len(background_files) == 1:
        print(f"[INFO] Only one background track found ({background_files[0]}). Skipping merge.")
        # Note: We should technically copy this file to output_path if we want
        # a consistent file name, but for this pipeline, just returning the
        # existing path is more efficient.
        return background_files[0]

    print(f"[INFO] Merging {len(background_files)} background tracks into {output_path}...")

    # 3. Construct the ffmpeg command
    cmd = ['ffmpeg', '-y'] # -y to overwrite output file

    # Add all background files as inputs
    for f in background_files:
        cmd.extend(['-i', f])

    # Create the filter_complex string to mix all inputs
    # e.g., "[0:a][1:a][2:a]amix=inputs=3[aout]"
    filter_inputs = "".join([f"[{i}:a]" for i in range(len(background_files))])
    filter_str = f"{filter_inputs}amix=inputs={len(background_files)}[aout]"

    cmd.extend(['-filter_complex', filter_str])

    # Map the mixed audio stream to the output file
    cmd.extend(['-map', '[aout]', output_path])

    # 4. Execute the command
    try:
        # capture_output=True hides ffmpeg's console spam unless there's an error
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print("[INFO] Background tracks merged successfully.")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg background merge failed.")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("[ERROR] FFmpeg not found. Please ensure it is installed and in your system's PATH.")
        return None

# ---
# 5. Supporting Module: Noise Analysis
# ---
def analyze_background_noise(background_audio_path: str) -> float:
    print(f"[INFO] Analyzing noise level for '{background_audio_path}'...")
    try:
        y, sr = librosa.load(background_audio_path, sr=None)
        rms = librosa.feature.rms(y=y)[0]
        mean_rms = float(np.mean(rms))
        print(f"[SUCCESS] Mean noise level (RMS) computed: {mean_rms:.4f}")
        return mean_rms
    except Exception as e:
        print(f"[ERROR] Could not analyze noise: {e}")
        return 0.0

# --- 3. MAIN ORCHESTRATOR ---
def run_preprocessing_pipeline(video_path: str) -> dict:
    """
    Runs the entire preprocessing pipeline for a single video.
    This is a modified version of your notebook's cell 8 function.
    """
    
    # We use a temporary directory for all outputs
    with tempfile.TemporaryDirectory() as base_output_dir:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(base_output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        audio_temp_path = os.path.join(video_output_dir, "extracted_audio.wav")
        separation_output_dir = os.path.join(video_output_dir, "audio_separation")

        final_outputs = {
            "transcribed_text": "",
            "base64_frames": [], # Renamed from 'frame_paths' for clarity
            "noise_level": 0.0,
        }

        # 1. Extract Video Frames
        final_outputs["base64_frames"] = extract_video_frames(video_path, interval_sec=2)

        # 2. Extract Audio
        if not extract_audio_ffmpeg(video_path, audio_temp_path):
            print("[WARNING] Audio extraction failed. Returning partial data.")
            return final_outputs

        # 3. Separate Vocals
        separation_paths = separate_vocals_demucs(audio_temp_path, separation_output_dir)
        if not separation_paths:
            print("[WARNING] Audio separation failed. Returning partial data.")
            return final_outputs

        raw_vocal_path = separation_paths.get('vocals')
        if not raw_vocal_path:
            print("[WARNING] No 'vocals' track found. Aborting audio processing.")
            return final_outputs

        # 4. Transcribe Vocals
        final_outputs["transcribed_text"] = transcribe_vocals(raw_vocal_path)

        # 5. Merge Background Tracks
        demucs_base_dir = os.path.dirname(raw_vocal_path)
        merged_background_path_out = os.path.join(demucs_base_dir, "merged_background.wav")
        merged_path = merge_background_tracks(separation_paths, merged_background_path_out)

        # 6. Analyze Background Noise
        if merged_path and os.path.exists(merged_path):
            final_outputs["noise_level"] = analyze_background_noise(merged_path)
        else:
            print("[WARNING] No background track available for noise analysis.")

        # 7. Return the data
        # The temporary directory and all its contents are automatically deleted
        
        print(f"[SUCCESS] Pipeline complete for {video_path}")
        return final_outputs