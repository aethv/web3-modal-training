# model_setup.py
import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import json
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_setup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Web3ModelSetup:
    def __init__(self, config_path="config.json"):
        """Initialize the model setup with configuration"""
        logger.info("Initializing Web3ModelSetup")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create necessary directories
        self._create_directories()
        
        # Check GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
    
    def _create_directories(self):
        """Create necessary directories for training"""
        dirs = [
            self.config["data_dir"],
            self.config["model_dir"],
            self.config["output_dir"],
            self.config["feedback_dir"]
        ]
        
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"Created directory: {dir_path}")
    
    def download_base_model(self):
        """Download the pre-trained model and tokenizer"""
        logger.info(f"Downloading base model: {self.config['base_model_name']}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['base_model_name'],
                trust_remote_code=True
            )
            
            # Configure tokenizer for generation
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['base_model_name'],
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            # Move model to appropriate device
            self.model.to(self.device)
            
            logger.info("Base model and tokenizer downloaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error downloading base model: {e}")
            return False
    
    def save_model_and_tokenizer(self):
        """Save the initialized model and tokenizer"""
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer not initialized")
            return False
        
        try:
            model_path = os.path.join(self.config["model_dir"], "base_model")
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            logger.info(f"Model and tokenizer saved to {model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving model and tokenizer: {e}")
            return False
    
    def prepare_initial_dataset(self):
        """Prepare the initial dataset for training"""
        logger.info("Preparing initial dataset")
        
        try:
            # Check if we should use a pre-existing dataset
            if self.config.get("use_existing_dataset", False):
                logger.info(f"Loading existing dataset from {self.config['existing_dataset_path']}")
                dataset = load_dataset(self.config['existing_dataset_path'])
                
            # Otherwise build dataset from scratch
            else:
                logger.info("Building dataset from web3 code samples")
                dataset = self._build_dataset_from_samples()
            
            # Save dataset for future use
            dataset_path = os.path.join(self.config["data_dir"], "initial_dataset")
            dataset.save_to_disk(dataset_path)
            logger.info(f"Dataset saved to {dataset_path}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            return None
    
    def _build_dataset_from_samples(self):
        """Build a dataset from web3 code samples"""
        # This would be replaced with your actual data loading logic
        prompts = []
        codes = []
        
        # Example: Load from a structured directory or CSV file
        data_source = self.config["data_source_path"]
        
        if os.path.isfile(data_source) and data_source.endswith('.csv'):
            # Load from CSV
            df = pd.read_csv(data_source)
            prompts = df['prompt'].tolist()
            codes = df['code'].tolist()
        
        else:
            # Load from directory structure
            # Assuming each pair is in a directory with prompt.txt and code.sol files
            for sample_dir in os.listdir(data_source):
                dir_path = os.path.join(data_source, sample_dir)
                if os.path.isdir(dir_path):
                    try:
                        # Read prompt
                        with open(os.path.join(dir_path, "prompt.txt"), 'r') as f:
                            prompt = f.read().strip()
                        
                        # Read code (could be .sol, .js, etc.)
                        code_files = [f for f in os.listdir(dir_path) if f.endswith(('.sol', '.js', '.ts'))]
                        if code_files:
                            with open(os.path.join(dir_path, code_files[0]), 'r') as f:
                                code = f.read().strip()
                            
                            prompts.append(prompt)
                            codes.append(code)
                    except Exception as e:
                        logger.warning(f"Error loading sample from {dir_path}: {e}")
        
        # Create and return the dataset
        return Dataset.from_dict({"prompt": prompts, "code": codes})
    
    def tokenize_dataset(self, dataset):
        """Tokenize the dataset for training"""
        logger.info("Tokenizing dataset")
        
        if self.tokenizer is None:
            logger.error("Tokenizer not initialized")
            return None
        
        def tokenize_function(examples):
            # Format inputs based on the model's expected format
            inputs = [
                f"{self.config['prompt_prefix']}{prompt}{self.config['prompt_suffix']}{code}{self.config['eos_token']}"
                for prompt, code in zip(examples['prompt'], examples['code'])
            ]
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.config["max_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Set labels equal to input_ids for causal language modeling
            model_inputs["labels"] = model_inputs["input_ids"].clone()
            
            return model_inputs
        
        try:
            # Split dataset
            train_test_split = dataset.train_test_split(
                test_size=self.config["test_split_ratio"]
            )
            
            # Tokenize both splits
            tokenized_dataset = train_test_split.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Save tokenized dataset
            tokenized_path = os.path.join(self.config["data_dir"], "tokenized_dataset")
            tokenized_dataset.save_to_disk(tokenized_path)
            logger.info(f"Tokenized dataset saved to {tokenized_path}")
            
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Error tokenizing dataset: {e}")
            return None
    
    def setup_training_args(self):
        """Set up the training arguments"""
        logger.info("Setting up training arguments")
        
        try:
            training_args = TrainingArguments(
                output_dir=self.config["output_dir"],
                per_device_train_batch_size=self.config["batch_size"],
                per_device_eval_batch_size=self.config["eval_batch_size"],
                gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
                learning_rate=self.config["learning_rate"],
                num_train_epochs=self.config["num_epochs"],
                weight_decay=self.config["weight_decay"],
                save_total_limit=self.config["save_total_limit"],
                logging_steps=self.config["logging_steps"],
                eval_steps=self.config["eval_steps"],
                save_steps=self.config["save_steps"],
                evaluation_strategy="steps",
                load_best_model_at_end=True,
                fp16=torch.cuda.is_available(),
                report_to="tensorboard",
            )
            
            # Save training args to file
            training_args_path = os.path.join(self.config["output_dir"], "training_args.json")
            with open(training_args_path, 'w') as f:
                json.dump(training_args.to_dict(), f, indent=4)
            
            logger.info(f"Training arguments saved to {training_args_path}")
            return training_args
            
        except Exception as e:
            logger.error(f"Error setting up training arguments: {e}")
            return None
    
    def setup_complete(self):
        """Check if all setup steps are complete and ready for training"""
        checks = [
            self.model is not None,
            self.tokenizer is not None,
            os.path.exists(os.path.join(self.config["data_dir"], "tokenized_dataset")),
            os.path.exists(os.path.join(self.config["output_dir"], "training_args.json"))
        ]
        
        all_ready = all(checks)
        if all_ready:
            logger.info("All setup steps complete. Ready for training!")
        else:
            logger.warning("Setup incomplete. Some steps are missing.")
        
        return all_ready

# Example configuration file structure
def create_example_config():
    """Create an example configuration file"""
    config = {
        "base_model_name": "codellama/CodeLlama-7b-hf",  # or another code-focused model
        "data_dir": "./data",
        "model_dir": "./models",
        "output_dir": "./outputs",
        "feedback_dir": "./feedback",
        "data_source_path": "./raw_data",
        "existing_dataset_path": "",  # Leave empty if building from scratch
        "use_existing_dataset": False,
        "prompt_prefix": "### Instruction: ",
        "prompt_suffix": " ### Response: ",
        "eos_token": "</s>",
        "max_length": 2048,
        "test_split_ratio": 0.1,
        "batch_size": 2,
        "eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "weight_decay": 0.01,
        "save_total_limit": 3,
        "logging_steps": 100,
        "eval_steps": 500,
        "save_steps": 500
    }
    
    with open("config_example.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    return config

# Main execution
if __name__ == "__main__":
    # Create example config if it doesn't exist
    if not os.path.exists("config.json"):
        config = create_example_config()
        with open("config.json", 'w') as f:
            json.dump(config, f, indent=4)
        logger.info("Created example config.json")
    
    # Initialize setup
    setup = Web3ModelSetup()
    
    # Download base model
    if setup.download_base_model():
        setup.save_model_and_tokenizer()
    
    # Prepare dataset
    dataset = setup.prepare_initial_dataset()
    if dataset is not None:
        tokenized_dataset = setup.tokenize_dataset(dataset)
    
    # Setup training arguments
    training_args = setup.setup_training_args()
    
    # Check if setup is complete
    is_ready = setup.setup_complete()
    logger.info(f"Setup ready for training: {is_ready}")