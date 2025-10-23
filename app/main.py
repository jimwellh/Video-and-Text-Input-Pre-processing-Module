# app/main.py
import tempfile
import shutil
import os  # <--- THIS IS THE FIX
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from app import pipeline
from app.models import PreprocessingResponse # Import our new model
# Make sure to import ConfigDict if you fixed the pydantic warning
# from app.models import PreprocessingResponse, ConfigDict 

# --- Lifespan Event ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    print("[INFO] Service starting up...")
    pipeline.load_model()
    yield
    # Clean up (if needed) on shutdown
    print("[INFO] Service shutting down...")

# Initialize the FastAPI app with the lifespan event
app = FastAPI(
    title="VRLook Video Pre-processing Service",
    description="Consumes a video file and returns transcription, frames, and noise level.",
    lifespan=lifespan
)

@app.post("/preprocess-video/", response_model=PreprocessingResponse)
async def preprocess_video(file: UploadFile = File(..., description="The house tour video file to process.")):
    """
    This endpoint matches your diagram's "Video and Text Input Pre-processing Module".
    
    It accepts a video file and returns the three required outputs for the next module:
    1.  **Text Input** (as `transcribed_text`)
    2.  **Image Input** (as a list of `base64_frames`)
    3.  **Noise Level Data** (as `noise_level`)
    """
    
    # Create a temporary file to store the upload
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_video_file:
            # Save the uploaded file to the temp file
            shutil.copyfileobj(file.file, temp_video_file)
            temp_video_path = temp_video_file.name
        
        print(f"[INFO] Received file: {file.filename}. Saved to {temp_video_path}")

        # --- Run the pipeline ---
        # This is the orchestrator function you just moved to pipeline.py
        results = pipeline.run_preprocessing_pipeline(temp_video_path)

        # Clean up the temporary video file
        os.remove(temp_video_path)

        print(f"[INFO] Processing complete. Returning results for {file.filename}")
        
        # Return the results, which FastAPI will validate against PreprocessingResponse
        return results

    except Exception as e:
        # Clean up the temp file if an error occurs
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        print(f"[ERROR] An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    finally:
        # Ensure the file object is closed
        file.file.close()

@app.get("/health")
def health_check():
    # A simple endpoint to check if the service is running
    return {"status": "ok", "asr_model_loaded": pipeline.ASR_PIPELINE is not None}