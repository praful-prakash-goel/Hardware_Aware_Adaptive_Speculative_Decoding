import os
import time
from typing import Optional

import torch
from model.model_architecture import build_model
from model.config import MAIN_MODEL_CONFIG, DRAFT_MODEL_SMALL_CONFIG, DRAFT_MODEL_MEDIUM_CONFIG, ModelConfig
from data.prepare_data import tokenizer
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    LogitsProcessor, LogitsProcessorList
)
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


def reset_cache(model):
    # custom model
    if hasattr(model, "blocks"):
        for block in model.blocks:
            if hasattr(block, "sa_heads"):
                block.sa_heads.reset_cache()
        return

    # huggingface model
    return


class _StepCallbackProcessor(LogitsProcessor):
    """Internal adapter for an optional per-token generation callback."""

    def __init__(self, callback, prompt_length):
        self.callback = callback
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        generated_tokens = input_ids.shape[1] - self.prompt_length
        self.callback(generated_tokens)
        return scores


def generate(model,
             input_ids,
             attention_mask,
             tokenizer=None,
             max_new_tokens=512,
             use_cache=True,
             temperature: float = 0.0,
             do_sample: bool = False,
             top_p: Optional[float] = None,
             repetition_penalty: Optional[float] = None,
             runtime_load_callback=None
             ):
    """
    Generate the output tokens based on the given prompt

    Args:
        model: Model to use for generation of the tokens
        input_ids: Input sequence of shape: (B, T)
        attention_mask: Mask for specifying tokens to attend
        tokenizer: Tokenizer to get token ids
        max_new_tokens: Maximum number of tokens to generate
        use_cache: Boolean variable to determine whether to use cache or not
        temperature: Sampling temperature (controls the creativity of the model)
        do_sample: If True sample, else greedy argmax
        top_p: If sampling and top_p provided, apply top_p (nucleus) sampling (controls the diversity of sampling)
        repetition_penalty: If provided penalizes repeated tokens (> 1.0)
        runtime_load_callback: Optional callback receiving generated-token count
                               once per decoding step.
    """

    if model is None:
        print(f"No model found, either checkpoint doesn't exist or the model is not passed as the parameter.")
        sys.exit()

    if tokenizer is None:
        print(f"Please provide an appropriate tokenizer for the model.")
        sys.exit()

    # If model is using cache, then reset cache before generation
    reset_cache(model)

    # Generate output for the given input ids
    model.eval()

    # Keep normal generation unchanged when no callback is supplied.
    logits_processors = None
    if runtime_load_callback is not None:
        runtime_processor = _StepCallbackProcessor(
            runtime_load_callback,
            prompt_length=input_ids.shape[1]
        )
        logits_processors = LogitsProcessorList([runtime_processor])

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
            eos_token_id=None,
            logits_processor=logits_processors
        )

    return output


if __name__ == '__main__':
    # CLI Arguments
    parser = argparse.ArgumentParser(description="Generate Function")

    parser.add_argument(
        "--model", type=str, default="main",
        choices=["main", "draft_small", "draft_medium", "pythia-1B", "pythia-160M", "SmolLM-1.7B", "SmolLM-135M",
                 "SmolLM2-1.7B", "SmolLM2-135M"], help="Select model to use for generation"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Disable KV cache"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--do_sample", action="store_true", help="Do sample"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Top p probability"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="Repetition penalty"
    )
    args = parser.parse_args()

    model_name = args.model
    max_new_tokens = args.max_new_tokens
    use_cache = not args.no_cache
    temperature = args.temperature
    do_sample = args.do_sample
    top_p = args.top_p
    repetition_penalty = args.repetition_penalty

    # Load the model and generate the output
    model, model_tokenizer = get_model(model_name=model_name)

    prompt = input("Please enter the prompt: ")
    inputs = model_tokenizer(prompt, return_tensors="pt")

    input_ids = inputs.input_ids.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)

    start = time.time()
    output = generate(model=model,
                      input_ids=input_ids,
                      attention_mask=attention_mask,
                      tokenizer=model_tokenizer,
                      max_new_tokens=max_new_tokens,
                      use_cache=use_cache,
                      temperature=temperature,
                      do_sample=do_sample,
                      top_p=top_p,
                      repetition_penalty=repetition_penalty
                      )

    text = model_tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print(f">> Output: {text}")
    print(f">> Time taken: {time.time() - start}")