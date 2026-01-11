import os
import io
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from huggingface_hub import AsyncInferenceClient
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

# Model Configuration
VISION_MODEL = "microsoft/Florence-2-large"
LOGIC_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Global client variable
hf_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    Initializes the Hugging Face AsyncInferenceClient.
    """
    global hf_client
    if not HF_API_KEY:
        logger.warning("HF_API_KEY not found in environment variables. Functionality will be limited.")
    
    hf_client = AsyncInferenceClient(token=HF_API_KEY)
    logger.info("Hugging Face client initialized.")
    yield
    # Cleanup if necessary (AsyncInferenceClient doesn't require explicit close, but good practice for other resources)
    logger.info("Shutting down application.")


app = FastAPI(title="Tag Team Image Analysis API", lifespan=lifespan)


@app.post("/analyze", response_class=JSONResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Endpoint to analyze an uploaded image using a two-step 'Tag Team' process:
    1. Vision Model (The Eye): Extracts text/description from the image.
    2. Logic Model (The Brain): Answers a query based on the image description.
    """
    if not HF_API_KEY:
         raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: HF_API_KEY is missing."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload an image."
        )

    try:
        # Read image content
        image_content = await file.read()
        
        # --- Step 1: The Vision Model (The Eye) ---
        logger.info(f"Sending image to Vision Model: {VISION_MODEL}")
        
        # Florence-2 works best with specific task prompts. 
        # For general captioning/OCR we can use '<MORE_DETAILED_CAPTION>' or '<OCR>' depending on the need.
        # The prompt implies we want to extracting text OR describe. 
        # Let's try to get a detailed description which covers both scenarios reasonably well for this context
        # or we could specifically ask for OCR if it's text-heavy.
        # Given "extract text or describe", let's use a task that does generally well.
        # However, the InferenceClient for image-to-text usually just takes the image.
        # For Florence-2 specifically via API, it's often good to send a text prompt if supported,
        # but the simple image_to_text call is the standard interface.
        
        try:
             vision_result = await hf_client.image_to_text(
                image=image_content,
                model=VISION_MODEL
            )
             # The output format for image_to_text models can vary. 
             # Usually it's a generated text string.
             # Florence-2 might return a JSON string or plain text depending on the task.
             # Default inference behavior usually returns the `generated_text`.
             
             image_description = vision_result.strip()
             logger.info(f"Vision Model Output: {image_description}")

        except Exception as e:
            logger.error(f"Vision Model failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail=f"Vision model service failed: {str(e)}"
            )

        # --- Step 2: The Logic Model (The Brain) ---
        prompt = (
            f"User uploaded an image containing this text: {image_description}. "
            f"Act as a helpful assistant and answer the user's request based on this."
        )
        
        logger.info(f"Sending prompt to Logic Model: {LOGIC_MODEL}")

        messages = [
            {"role": "user", "content": prompt}
        ]

        try:
            # Qwen/Qwen2.5-3B-Instruct is a text-generation model
            response = await hf_client.chat_completion(
                model=LOGIC_MODEL,
                messages=messages,
                max_tokens=500, # Reasonable limit for an answer
                temperature=0.7
            )
            
            final_answer = response.choices[0].message.content
            logger.info("Logic Model response received.")

        except Exception as e:
             logger.error(f"Logic Model failed: {e}")
             raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail=f"Logic model service failed: {str(e)}"
            )

        return {
            "vision_output": image_description,
            "final_answer": final_answer
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred processing your request."
        )
