# model_api.py
import os
import torch
import logging
import json
import pandas as pd

from fastapi import FastAPI, BackgroundTasks, HTTPException, Body
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Any, Optional

from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# FastAPI app
app = FastAPI(title="Web3 App Generator API")

# Model storage
class ModelManager:
    def __init__(self, model_path=None):
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load model and tokenizer from path"""
        logger.info(f"Loading model from {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # Create generator pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1
            )
            
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def generate(self, prompt, max_length=2048, temperature=0.7, top_p=0.9, top_k=50):
        """Generate code based on prompt"""
        if not self.generator:
            raise ValueError("Model not loaded")
        
        # Format prompt according to model's expected format
        formatted_prompt = f"{config['prompt_prefix']}{prompt}{config['prompt_suffix']}"
        
        try:
            result = self.generator(
                formatted_prompt,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated code
            generated_text = result[0]["generated_text"]
            
            # Remove the prompt part
            response_start = generated_text.find(config['prompt_suffix']) + len(config['prompt_suffix'])
            code = generated_text[response_start:].strip()
            
            # Remove any EOS tokens that might be in the text
            code = code.replace(config['eos_token'], "").strip()
            
            return code
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise

# Initialize model manager
model_manager = ModelManager()

# Data models
class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

class GenerationResponse(BaseModel):
    code: str
    generation_time: float

class FeedbackRequest(BaseModel):
    prompt: str
    generated_code: str
    corrected_code: str
    feedback: Optional[str] = None
    rating: Optional[int] = None

class TrainingStatusResponse(BaseModel):
    status: str
    last_update: str
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    loss: Optional[float] = None

# Feedback storage
class FeedbackManager:
    def __init__(self, feedback_dir):
        self.feedback_dir = feedback_dir
        if not os.path.exists(feedback_dir):
            os.makedirs(feedback_dir)
    
    def save_feedback(self, feedback_data):
        """Save user feedback for future training"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feedback_id = f"feedback_{timestamp}"
        
        # Store the feedback data
        feedback_path = os.path.join(self.feedback_dir, f"{feedback_id}.json")
        with open(feedback_path, "w") as f:
            json.dump(feedback_data, f, indent=2)
        
        logger.info(f"Feedback saved to {feedback_path}")
        return feedback_id
    
    def prepare_training_data(self):
        """Convert collected feedback into training format"""
        all_feedbacks = []
        
        for filename in os.listdir(self.feedback_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.feedback_dir, filename), "r") as f:
                    feedback = json.load(f)
                    all_feedbacks.append(feedback)
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(all_feedbacks)
        
        # Save as CSV for training
        output_path = os.path.join(self.feedback_dir, "combined_feedback.csv")
        df.to_csv(output_path, index=False)
        
        logger.info(f"Prepared training data with {len(df)} feedback entries")
        return output_path, len(df)

# Initialize feedback manager
feedback_manager = FeedbackManager(config["feedback_dir"])

# Training status tracker
class TrainingStatusTracker:
    def __init__(self):
        self.status = "idle"  # idle, preparing, training, completed, failed
        self.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.current_epoch = None
        self.total_epochs = None
        self.loss = None
    
    def update_status(self, status, current_epoch=None, total_epochs=None, loss=None):
        self.status = status
        self.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if current_epoch is not None:
            self.current_epoch = current_epoch
        
        if total_epochs is not None:
            self.total_epochs = total_epochs
        
        if loss is not None:
            self.loss = loss
    
    def get_status(self):
        return {
            "status": self.status,
            "last_update": self.last_update,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "loss": self.loss
        }

# Initialize training status tracker
training_status = TrainingStatusTracker()

# Asynchronous training task
async def train_from_feedback(background_tasks):
    """Train model from feedback in background"""
    try:
        # Update status
        training_status.update_status("preparing")
        
        # Prepare training data
        data_path, num_examples = feedback_manager.prepare_training_data()
        
        # Check if we have enough examples
        min_examples = config.get("training_examples_required", 100)
        if num_examples < min_examples:
            logger.warning(f"Not enough training examples: {num_examples}/{min_examples}")
            training_status.update_status("failed", 
                                          loss=f"Not enough examples: {num_examples}/{min_examples}")
            return False
        
        # Update status
        training_status.update_status("training", total_epochs=config["num_epochs"])
        
        # Import training module here to prevent circular imports
        from train import train_model
        
        # Train the model
        final_model_path = train_model(
            "config.json",
            resume=False,
            checkpoint_path=None
        )
        
        # Update model manager with new model
        if model_manager.load_model(final_model_path):
            training_status.update_status("completed")
            return True
        else:
            training_status.update_status("failed", loss="Failed to load updated model")
            return False
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        training_status.update_status("failed", loss=str(e))
        return False

# API Routes
@app.post("/generate", response_model=GenerationResponse)
async def generate_code(request: GenerationRequest):
    """Generate web3 code from prompt"""
    if not model_manager.model:
        # Try to load the latest model
        latest_model_path = os.path.join(config["output_dir"], "final_model")
        if not os.path.exists(latest_model_path):
            latest_model_path = os.path.join(config["model_dir"], "base_model")
        
        if not model_manager.load_model(latest_model_path):
            raise HTTPException(status_code=500, detail="No model is loaded")
    
    try:
        import time
        start_time = time.time()
        
        code = model_manager.generate(
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        generation_time = time.time() - start_time
        
        return {"code": code, "generation_time": generation_time}
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for model improvement"""
    try:
        feedback_data = {
            "prompt": request.prompt,
            "generated_code": request.generated_code,
            "corrected_code": request.corrected_code,
            "feedback": request.feedback,
            "rating": request.rating,
            "timestamp": datetime.now().isoformat()
        }
        
        feedback_id = feedback_manager.save_feedback(feedback_data)
        return {"status": "success", "feedback_id": feedback_id}
    
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    """Trigger model retraining from feedback"""
    if training_status.status in ["preparing", "training"]:
        return {"status": "error", "message": "Training already in progress"}
    
    background_tasks.add_task(train_from_feedback, background_tasks)
    
    return {"status": "success", "message": "Training triggered"}

@app.get("/training-status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Get current training status"""
    return training_status.get_status()

@app.get("/load-model")
async def load_specific_model(model_path: str):
    """Load a specific model"""
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")
    
    success = model_manager.load_model(model_path)
    if success:
        return {"status": "success", "message": f"Model loaded from {model_path}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    try:
        # Try to load the final model first
        final_model_path = os.path.join(config["output_dir"], "final_model")
        if os.path.exists(final_model_path):
            model_manager.load_model(final_model_path)
        else:
            # Fall back to the base model
            base_model_path = os.path.join(config["model_dir"], "base_model")
            if os.path.exists(base_model_path):
                model_manager.load_model(base_model_path)
            else:
                logger.warning("No model found. Please initialize a model first.")
    except Exception as e:
        logger.error(f"Startup error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)