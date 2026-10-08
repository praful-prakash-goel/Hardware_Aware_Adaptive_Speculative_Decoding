import torch
import time
import pandas as pd
from inference.generate import get_model
from inference.speculative_engine import generate_speculative_custom, generate_speculative_standard
from functools import partial
import argparse
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results/")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Defining the prompts to use for benchmarking
PROMPTS = [
    "The capital of France is",
    "Artificial Intelligence works by",
    "The history of the Roman Empire is vast and"
]

def fix_pad(model, tokenizer):
    # Ensure eos token id is int
    eos_id = model.generation_config.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]
    
    # Fix tokenizer pad token id    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure pad token id is int
    if isinstance(tokenizer.pad_token_id, list):
        tokenizer.pad_token_id = tokenizer.pad_token_id[0]
    
    # Set everything
    model.config.pad_token_id = tokenizer.pad_token_id or eos_id
    model.generation_config.pad_token_id = model.config.pad_token_id

    # Disable EOS id
    model.generation_config.eos_token_id = None
    
def calculate_tps(generate_func,
                  max_new_tokens,
                  use_cache,
                  model_tokenizer,
                  method_name=None,
                  device=DEVICE,
                  prompts=PROMPTS,
                  reset_callback=None,
                  verbose=False,
                  controller=None,
                  controller_main_model=None,
                  controller_draft_model=None,
                  include_controller_init=True,
                  ):
    """
    Calculates average tokens per second for the given model

    Args:
        generate_func: Function to call for token generation
        max_new_tokens: Maximum number of tokens to generate
        use_cache: Boolean variable to determine whether to use cache or not
        method_name: Name of the method for which tps is being calculated
        device: Device to use for token generation
        prompts: List of prompts on which the benchmarks will be calculated
        reset_callback: if use_cache is True, then call reset_callback() to reset cache before generation
        verbose: if True, then print debugging statements
        controller: Optional AdaptiveController prototype.
        controller_main_model: Main model passed to controller.initialize().
        controller_draft_model: Draft model passed to controller.initialize().
        include_controller_init:
            If True, controller initialization is included in the
            measured end-to-end time.
            If False, only generation time is measured.
    """
    
    timings = []
    tokens_generated = []
    acceptance_list = []
    mean_accepted_list = []
    
    # Warmup the model
    for _ in range(2):
        inputs = model_tokenizer("This is a warmup", return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        if reset_callback is not None:
            reset_callback()
        
        with torch.no_grad():
            _ = generate_func(input_ids=input_ids, max_new_tokens=max_new_tokens, use_cache=use_cache, attention_mask=attention_mask)

        if device == "cuda":
            torch.cuda.synchronize()

    # Actual Test
    if verbose and method_name:
        print(f"\n>> Running benchmark for {method_name}...")
        
    for p in prompts:
        inputs = model_tokenizer(p, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        if reset_callback is not None:
            reset_callback()

        active_controller = None
        sd_viable = True
        init_time = 0.0

        if controller is not None:

            if controller_main_model is None:
                raise ValueError(
                    "controller_main_model must be provided "
                    "when controller is used."
                )

            if controller_draft_model is None:
                raise ValueError(
                    "controller_draft_model must be provided "
                    "when controller is used."
                )

            # Create a fresh controller with the same settings
            active_controller = type(controller)(
                alpha=controller.alpha,
                eval_every=controller.eval_every,
                re_explore_every=controller.re_explore_every
            )

            # Initialize using THIS prompt
            if include_controller_init:
                if device == "cuda":
                    torch.cuda.synchronize()

                init_start = time.time()

                with torch.no_grad():
                    sd_viable = active_controller.initialize(
                        main_model=controller_main_model,
                        draft_model=controller_draft_model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        device=device,
                        use_cache=use_cache
                    )

                if device == "cuda":
                    torch.cuda.synchronize()

                init_time = time.time() - init_start

            else:
                # Initialization happens outside measured generation time
                with torch.no_grad():
                    sd_viable = active_controller.initialize(
                        main_model=controller_main_model,
                        draft_model=controller_draft_model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        device=device,
                        use_cache=use_cache
                    )

        # Start timer
        if device == "cuda": torch.cuda.synchronize()
        start_time = time.time()

        # If initialization should be included, move the start
        # backwards so total duration = init + generation.
        if controller is not None and include_controller_init:
            start_time -= init_time

        with torch.no_grad():
            if controller is not None and sd_viable:
                output = generate_func(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                    controller=active_controller
                )

            elif controller is not None and not sd_viable:
                # Controller decided SD is not worthwhile
                # -> use normal AR generation
                output = controller_main_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                    eos_token_id=None
                )

            else:
                # Normal benchmark: AR or vanilla SD
                output = generate_func(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    use_cache=use_cache,
                    attention_mask=attention_mask
                )

        # Stop timer
        if device == "cuda": torch.cuda.synchronize()
        end_time = time.time()

        if isinstance(output, tuple):
            output_ids, stats = output

            if "acceptance_rate" in stats:
                acceptance_list.append(
                    stats["acceptance_rate"]
                )

            if "mean_accepted" in stats:
                mean_accepted_list.append(
                    stats["mean_accepted"]
                )
        else:
            output_ids = output


        # Calculate time taken and number of tokens generated
        duration = end_time - start_time
        total_len = output_ids.shape[1]
        input_len = input_ids.shape[1]
        generated_tokens = total_len - input_len
        
        timings.append(duration)
        tokens_generated.append(generated_tokens)
        
        del output, output_ids
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Calculate avg tps
    total_time = sum(timings)
    total_tokens = sum(tokens_generated)

    # calculate avg acceptance and mean accepted for speculative engine
    avg_tps = total_tokens / total_time if total_time > 0 else 0.0
    if any(a is not None for a in acceptance_list):
        avg_acceptance = sum(acceptance_list) / len(acceptance_list)
        avg_mean_accepted = sum(mean_accepted_list) / len(mean_accepted_list)
    else:
        avg_acceptance = None
        avg_mean_accepted = None
    
    return avg_tps, avg_acceptance, avg_mean_accepted

# Callback to reset cache
def reset_main():
    if model == "custom":
        for block in main_model.blocks:
            block.sa_heads.reset_cache()

def reset_draft():
    if model == "custom":
        for draft in draft_models.values():
            for block in draft[0].blocks:
                block.sa_heads.reset_cache()

def reset_both():
    if model == "custom":
        reset_draft()
        reset_main()
    else:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    print(f">> Benchmarking on: {DEVICE.upper()}\n")
    # CLI Arguments
    parser = argparse.ArgumentParser("Benchmark tps of different configurations")

    parser.add_argument(
        "--model", type=str, default="custom",
        choices=["custom", "pythia", "SmolLM", "SmolLM2"], help="Model family to perform benchmark"
    )
    parser.add_argument(
        "--gamma", type=int, default=5, help="Number of draft tokens to speculate per step"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate"
    )
    args = parser.parse_args()
    
    model = args.model
    gamma = args.gamma
    max_new_tokens = args.max_new_tokens
    
    print("----- Running the benchmarks -----\n")
    
    # Load the model
    if model == "custom":
        SAVE_PATH = os.path.join(RESULTS_DIR, "benchmarks.csv")
        STRESS_PATH = os.path.join(RESULTS_DIR, "stress_test.csv")
        
        print(f">> Loading {model} Models...")
        main_model, main_tokenizer = get_model(model_name="main")
        draft_models = {
            "small": get_model(model_name="draft_small"),
            "medium": get_model(model_name="draft_medium")
        }
        stress_draft_name = "Medium"
        main_model_name = "Custom"
        
    elif model == "pythia":
        SAVE_PATH = os.path.join(RESULTS_DIR, "benchmarks_pythia.csv")
        STRESS_PATH = os.path.join(RESULTS_DIR, "stress_test_pythia.csv")
        
        print(f">> Loading {model} models...")
        main_model, main_tokenizer = get_model(model_name="pythia-1B")
        fix_pad(main_model, main_tokenizer)
        
        draft_model, draft_tokenizer = get_model(model_name="pythia-160M")
        fix_pad(draft_model, draft_tokenizer)
        draft_models = {
            "pythia-160M": (draft_model, draft_tokenizer)
        }
        stress_draft_name = "pythia-160M"
        main_model_name = "pythia-1B"
    
    elif model == "SmolLM":
        SAVE_PATH = os.path.join(RESULTS_DIR, "benchmarks_smol.csv")
        STRESS_PATH = os.path.join(RESULTS_DIR, "stress_test_smol.csv")
        
        print(f">> Loading {model} models...")
        main_model, main_tokenizer = get_model(model_name="SmolLM-1.7B")
        fix_pad(main_model, main_tokenizer)
        
        draft_model, draft_tokenizer = get_model(model_name="SmolLM-135M")
        fix_pad(draft_model, draft_tokenizer)
        draft_models = {
            "smollm-135M": (draft_model, draft_tokenizer)
        }
        stress_draft_name = "smollm-135M"
        main_model_name = "smollm-1.7B"
        
    elif model == "SmolLM2":
        SAVE_PATH = os.path.join(RESULTS_DIR, "benchmarks_smol2.csv")
        STRESS_PATH = os.path.join(RESULTS_DIR, "stress_test_smol2.csv")
        
        print(f">> Loading {model} models...")
        main_model, main_tokenizer = get_model(model_name="SmolLM2-1.7B")
        fix_pad(main_model, main_tokenizer)
        
        draft_model, draft_tokenizer = get_model(model_name="SmolLM2-135M")
        fix_pad(draft_model, draft_tokenizer)
        draft_models = {
            "smollm2-135M": (draft_model, draft_tokenizer)
        }
        stress_draft_name = "smollm2-135M"
        main_model_name = "smollm2-1.7B"
        
    else:
        print(f"\n>> Error: Unknown model setup. Supported models: custom, pythia, SmolLM and SmolLM2")
        exit()
    
    if main_model and all(draft_models.values()):
        # results = []
        #
        # main_model.eval()
        # for draft in draft_models.values():
        #     draft[0].eval()
        #
        # print("\n===== BASELINE TPS =====")
        # header = f"{'Model':<15} {'No Cache':>12} {'Cache':>12}"
        # print(header)
        # print(f"-" * len(header))
        #
        # # Calculate avg tps for main model
        # tps_main_without_cache, _, _ = calculate_tps(generate_func=main_model.generate, max_new_tokens=max_new_tokens, method_name="main without cache", use_cache=False, model_tokenizer=main_tokenizer)
        # tps_main_with_cache, _, _ = calculate_tps(generate_func=main_model.generate, max_new_tokens=max_new_tokens, method_name="main with cache", use_cache=True, model_tokenizer=main_tokenizer, reset_callback=reset_main)
        # print(f"{f'Main {main_model_name}':<15} {tps_main_without_cache:>12.2f} {tps_main_with_cache:>12.2f}")
        # time.sleep(10)
        #
        # # Store Main Results
        # results.extend([
        #     {"method": f"Main {main_model_name}", "draft": None, "gamma": None, "cache": False, "tps": tps_main_without_cache, "speedup": None, "acceptance": None, "mean_accepted": None},
        #     {"method": f"Main {main_model_name}", "draft": None, "gamma": None, "cache": True, "tps": tps_main_with_cache, "speedup": None, "acceptance": None, "mean_accepted": None}
        # ])
        #
        # for draft_name, draft_model in draft_models.items():
        #     # Calculate avg tps for draft small model
        #     tps_draft_without_cache, _, _ = calculate_tps(generate_func=draft_model[0].generate, max_new_tokens=max_new_tokens, method_name=f"draft {draft_name} without cache", use_cache=False, model_tokenizer=draft_model[1])
        #     tps_draft_with_cache, _, _ = calculate_tps(generate_func=draft_model[0].generate, max_new_tokens=max_new_tokens, method_name="draft small with cache", use_cache=True, model_tokenizer=draft_model[1], reset_callback=reset_draft)
        #     print(f"{f'Draft {draft_name}':<15} {tps_draft_without_cache:>12.2f} {tps_draft_with_cache:>12.2f}")
        #     time.sleep(10)
        #
        #     # Store Draft Results
        #     results.extend([
        #         {"method": f"Draft {draft_name}", "draft": draft_name, "gamma": None, "cache": False, "tps": tps_draft_without_cache, "speedup": None, "acceptance": None, "mean_accepted": None},
        #         {"method": f"Draft {draft_name}", "draft": draft_name, "gamma": None, "cache": True, "tps": tps_draft_with_cache, "speedup": None, "acceptance": None, "mean_accepted": None}
        #     ])
        #
        # # Calculate avg tps and speedup for speculative decoding
        # print(f"\n===== Speculative (gamma = {gamma}) =====")
        # header = f"{'Draft':<15} {'NoCache TPS':>12} {'Cache TPS':>12} {'NoCache x':>12} {'Cache x':>12}"
        # print(header)
        # print("-" * len(header))
        # for draft_name, draft_model in draft_models.items():
        #     if model == "custom":
        #         generate_func = partial(
        #             generate_speculative_custom,
        #             main_model,
        #             draft_model[0],
        #             gamma=gamma,
        #             return_stats=True
        #         )
        #     else:
        #         generate_func = partial(
        #             generate_speculative_standard,
        #             main_model,
        #             draft_model[0],
        #             gamma=gamma,
        #             return_stats=True
        #         )
        #
        #     tps_speculative_without_cache, _, _ = calculate_tps(generate_func=generate_func, max_new_tokens=max_new_tokens, method_name=f"speculative {draft_name} without cache", use_cache=False, model_tokenizer=main_tokenizer)
        #     tps_speculative_with_cache, _, _ = calculate_tps(generate_func=generate_func, max_new_tokens=max_new_tokens, method_name=f"speculative {draft_name} with cache", use_cache=True, model_tokenizer=main_tokenizer, reset_callback=reset_both)
        #     time.sleep(10)
        #
        #     speedup_without_cache = tps_speculative_without_cache / tps_main_without_cache
        #     speedup_with_cache = tps_speculative_with_cache / tps_main_with_cache
        #
        #     print(
        #         f"{draft_name:<15} "
        #         f"{tps_speculative_without_cache:>12.2f} "
        #         f"{tps_speculative_with_cache:>12.2f} "
        #         f"{speedup_without_cache:>12.2f} "
        #         f"{speedup_with_cache:>12.2f}"
        #     )
        #
        # # Perform gamma sweep
        # print(f"\n>> Performing gamma sweep for benchmarking...")
        # gamma_values = [1, 2, 3, 5, 7, 10]
        # for gamma in gamma_values:
        #     print(f">> Gamma: {gamma}")
        #     for draft_name, draft_model in draft_models.items():
        #         if model == "custom":
        #             generate_func = partial(
        #                 generate_speculative_custom,
        #                 main_model,
        #                 draft_model[0],
        #                 gamma=gamma,
        #                 return_stats=True
        #             )
        #         else:
        #             generate_func = partial(
        #                 generate_speculative_standard,
        #                 main_model,
        #                 draft_model[0],
        #                 gamma=gamma,
        #                 return_stats=True
        #             )
        #
        #         tps_without, acceptance_without, mean_accepted_without = calculate_tps(generate_func=generate_func, max_new_tokens=max_new_tokens, method_name=f"speculative {draft_name} without cache", use_cache=False, model_tokenizer=main_tokenizer)
        #         tps_with, acceptance_with, mean_accepted_with = calculate_tps(generate_func=generate_func, max_new_tokens=max_new_tokens, method_name=f"speculative {draft_name} with cache", use_cache=True, model_tokenizer=main_tokenizer, reset_callback=reset_both)
        #         time.sleep(10)
        #
        #         speedup_without = tps_without / tps_main_without_cache
        #         speedup_with = tps_with / tps_main_with_cache
        #
        #         # Store results for dataframe
        #         results.append({
        #             "method": "speculative",
        #             "draft": draft_name,
        #             "gamma": gamma,
        #             "cache": False,
        #             "tps": tps_without,
        #             "speedup": speedup_without,
        #             "acceptance": acceptance_without,
        #             "mean_accepted": mean_accepted_without
        #         })
        #
        #         results.append({
        #             "method": "speculative",
        #             "draft": draft_name,
        #             "gamma": gamma,
        #             "cache": True,
        #             "tps": tps_with,
        #             "speedup": speedup_with,
        #             "acceptance": acceptance_with,
        #             "mean_accepted": mean_accepted_with
        #         })
        #
        # df = pd.DataFrame(results)
        # df.to_csv(SAVE_PATH)
        # print(f"\n>> Benchmarks result saved at {SAVE_PATH}")
        
        print(f"\n>> Performing stress test for different context lengths...")
        stress_results = []
        stress_draft = draft_models[stress_draft_name][0]
        # TEMP START
        stress_df = pd.read_csv(STRESS_PATH)
        # TEMP ENDS
        
        for context_length in [1028, 2056]:
            print(f">> Context Length: {context_length}")
            
            tps_main_cache, _, _ = calculate_tps(generate_func=main_model.generate, max_new_tokens=context_length, use_cache=True, model_tokenizer=main_tokenizer, reset_callback=reset_main)
            time.sleep(10)
            stress_results.append({
                'context_length': context_length,
                "configuration": f"Main {main_model_name} (With Cache)",
                "tps": tps_main_cache
            })
            
            if model == "custom":
                generate_func = partial(
                    generate_speculative_custom,
                    main_model,
                    stress_draft,
                    gamma=5
                )
            else:
                generate_func = partial(
                    generate_speculative_standard,
                    main_model,
                    stress_draft,
                    gamma=5
                )
            
            # tps_speculative_without, _, _ = calculate_tps(generate_func=generate_func, max_new_tokens=context_length, use_cache=False, model_tokenizer=main_tokenizer)
            # time.sleep(10)
            # stress_results.append({
            #     "context_length": context_length,
            #     "configuration": f"Speculative {stress_draft_name} (Without Cache)",
            #     "tps": tps_speculative_without
            # })
            
            tps_speculative_with, _, _ = calculate_tps(generate_func=generate_func, max_new_tokens=context_length, use_cache=True, model_tokenizer=main_tokenizer, reset_callback=reset_both)
            time.sleep(10)
            stress_results.append({
                "context_length": context_length,
                "configuration": f"Speculative {stress_draft_name} (With Cache)",
                "tps": tps_speculative_with
            })

        # TEMP START
        new_df = pd.DataFrame(stress_results)
        stress_df = pd.concat([stress_df, new_df])
        # TEMP END

        # stress_df = pd.DataFrame(stress_results)
        stress_df.to_csv(STRESS_PATH)
        print(f"\n>> Stress Test result saved at {STRESS_PATH}")
        
    else:
        print(f"\n>> Error while loading models.")