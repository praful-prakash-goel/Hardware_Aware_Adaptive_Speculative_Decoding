import os
import torch
from model.model_architecture import build_model
from model.config import MAIN_MODEL_CONFIG, DRAFT_MODEL_SMALL_CONFIG, DRAFT_MODEL_MEDIUM_CONFIG, ModelConfig
from data.prepare_data import tokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import sys
import argparse

BASE_DIR = "saved_models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_config(model_name, config_path=None):
    '''
    Retrieves config for a specific model name
    
    Args:
        model_name: Name of the model for which the config has to be loaded
        config_path: Path where to look for the config file
    '''
    
    # Load config from json file if it exists
    if config_path and os.path.exists(config_path):
        print(f">> Loading config from artifact: {config_path}")
        return ModelConfig.from_json(config_path)

    # Fallback to hardcoded python object is json does not exists
    print(f">> JSON not found. Using hardcoded python config for {model_name} model...")
    if model_name == 'main':
        return MAIN_MODEL_CONFIG
    elif model_name == 'draft_small':
        return DRAFT_MODEL_SMALL_CONFIG
    elif model_name == 'draft_medium':
        return DRAFT_MODEL_MEDIUM_CONFIG
    else:
        raise ValueError(f">> Unknown model name: {model_name}")
        
def get_model(model_name, checkpoint_dir=BASE_DIR, device=DEVICE):
    '''
    Loads model by name: "main" or "draft"
    
    Args:
        model_name: Name of the model to load
        checkpoint_dir: Base dir from where the checkpoint will be loaded
        device: Device to use
    '''
    
    if model_name in ["main", "draft_small", "draft_medium"]:
        # Construct path dynamically based on the model requested
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_model.pt")
        config_path = os.path.join(checkpoint_dir, f"{model_name}_config.json")
        
        # Load the checkpoint of the requested model if exist
        if os.path.exists(checkpoint_path):
            # Load config
            config = get_config(model_name, config_path)
            
            # Build the model
            print(f">> Building {model_name} model...")
            model = build_model(device=device, config=config)
            
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            model_tokenizer = tokenizer
            model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
            
            return model, model_tokenizer
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    elif model_name == 'pythia-1B':
        print(f">> Building {model_name} model...")
        model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1b').to(device)
        model_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-1b')
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        
        return model, model_tokenizer
    elif model_name == 'pythia-160M':
        print(f">> Building {model_name} model...")
        model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m').to(device)
        model_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        
        return model, model_tokenizer
    elif model_name == 'SmolLM-1.7B':
        print(f">> Building {model_name} model...")
        model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM-1.7B').to(device)
        model_tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-1.7B')
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        
        return model, model_tokenizer
    elif model_name == 'SmolLM-135M':
        print(f">> Building {model_name} model...")
        model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM-135M').to(device)
        model_tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-1.7B')
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        
        return model, model_tokenizer
    elif model_name == 'SmolLM2-1.7B':
        print(f">> Building {model_name} model...")
        model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-1.7B').to(device)
        model_tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B')
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        
        return model, model_tokenizer
    elif model_name == 'SmolLM2-135M':
        print(f">> Building {model_name} model...")
        model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-135M').to(device)
        model_tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B')
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        
        return model, model_tokenizer
    elif model_name == 'SmolLM2-360M':
            print(f">> Building {model_name} model...")
            model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-360M').to(device)
            model_tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B')
            model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
            
            return model, model_tokenizer
    else:
        raise ValueError(f">> Unknown model name: {model_name}.")

def get_blocks(model):
    # Custom model architecture
    if hasattr(model, "blocks"):
        return model.blocks
    
    # GPT-2 family
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    
    # Meta OPT family
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    
    raise AttributeError("Model does not have recognizable transformer blocks")
    
def reset_cache(model):
    # custom model
    if hasattr(model, "blocks"):
        for block in model.blocks:
            if hasattr(block, "sa_heads"):
                block.sa_heads.reset_cache()
        return None

    # huggingface model
    return None

def generate(prompt = None, model = None, tokenizer = None, device=DEVICE, max_new_tokens=512, use_cache=True):
    '''
    Generate the output tokens based on the given prompt
    
    Args:
        prompt: Initial decoder tokens
        model: Model to use for generation of the tokens
        device: Device to use for generation
        max_new_tokens: Maximum number of tokens to generate
        use_cache: Boolean variable to determine whether to use cache or not
    '''
    
    if model == None:
        print(f"No model found, either checkpoint doesn't exist or the model is not passed as the parameter.")
        sys.exit()
    
    if tokenizer == None:
        print(f"Please provide an appropriate tokenizer for the model.")
        sys.exit()
    
    # If model is using cache, then reset cache before generation
    _ = reset_cache(model)
            
    # Take prompt as input if not already provided
    if prompt is None:
        prompt = input("Please enter the prompt: ")
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Generate output for the given input ids
    model.eval()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.5,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
            eos_token_id=None
        )[0].tolist()
    text = tokenizer.decode(output, skip_special_tokens=True)
    
    print(f"\n>> Output: {text}")
    
if __name__ == '__main__':
    # CLI Arguments
    parser = argparse.ArgumentParser(description="Generate Function")
    
    parser.add_argument(
        "--model", type=str, default="main",
        choices=["main", "draft_small", "draft_medium", "pythia-1B", "pythia-160M", "SmolLM-1.7B", "SmolLM-135M", "SmolLM2-1.7B", "SmolLM2-135M"], help="Select model to use for generation"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Disable KV cache"
    )
    args = parser.parse_args()
    
    model_name = args.model
    max_new_tokens = args.max_new_tokens
    use_cache = not args.no_cache
    
    # Load the model and generate the output
    model, model_tokenizer = get_model(model_name=model_name)
    generate(model=model, use_cache=use_cache, max_new_tokens=max_new_tokens, tokenizer=model_tokenizer)