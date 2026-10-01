import torch
import os
import argparse
import pandas as pd
from inference.generate import get_model
from experiments.benchmark_tps import fix_pad
from inference.speculative_engine import generate_speculative_standard

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results/")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Defining the prompts to use for benchmarking
PROMPTS = [
    "The capital of France is",
    "Artificial Intelligence works by",
    "The history of the Roman Empire is vast and",
    "The future of computer science relies completely on the fact that"
]

class SpeculativeProfiler:
    def __init__(self, device="cuda"):
        self.device = device
        self.is_cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.stages = [
            "draft_speculation",
            "target_verification",
            "comparison_logic",
            "cache_rollback",
            "tensor_updates"
        ]
        self.timings = {stage: 0.0 for stage in self.stages}
        self._active_events = {}
    
    def start_stage(self, stage_name: str):
        if self.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self._active_events[stage_name] = start_event
        
    def end_stage(self, stage_name: str):
        if self.is_cuda and stage_name in self._active_events:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            
            if not hasattr(self, "_recorded_pairs"):
                self._recorded_pairs = []
            self._recorded_pairs.append((stage_name, self._active_events.pop(stage_name), end_event))
    
    def finalize(self):
        if self.is_cuda and hasattr(self, "_recorded_pairs"):
            torch.cuda.synchronize()
            for stage_name, start_event, end_event in self._recorded_pairs:
                elapsed_ms = start_event.elapsed_time(end_event)
                self.timings[stage_name] += elapsed_ms
            self._recorded_pairs.clear()
        
        total_time = sum(self.timings.values())
        breakdown = {
            stage: {
                "total_ms": round(t, 2),
                "pct": round((t / total_time * 100), 2) if total_time > 0 else 0.0
            }
            for stage, t in self.timings.items()
        }
        
        return breakdown, round(total_time, 2)

def run_profiling_benchmark(main_model, draft_model, tokenizer, save_path, max_new_tokens, prompts=PROMPTS, gammas=[1,2,3,5,7,10], device=DEVICE):
    warmup_text = "This is a warmup"
    
    inputs = tokenizer(warmup_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    # Warm up GPU
    for _ in range(3):
        _ = main_model(input_ids, use_cache=False)
        _ = draft_model(input_ids, use_cache=False)
    torch.cuda.synchronize()
    
    results = []
    
    print(f"\n>> Starting Profiling across {len(prompts)} prompts...")
    
    for gamma in gammas:
        print(f"\n>> Gamma: {gamma}")
        for use_cache in [True, False]:
            total_times = []
            acceptance_rates = []
            mean_accepted_list = []
            stage_accumulators = {}
            
            for prompt in prompts:
                profiler = SpeculativeProfiler(device=device)
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                
                _, acceptance_rate, mean_accepted, breakdown, total_ms = generate_speculative_standard(
                    main_model=main_model,
                    draft_model=draft_model,
                    input_ids=input_ids.clone(),
                    tokenizer=tokenizer,
                    attention_mask=attention_mask.clone(),
                    max_new_tokens=max_new_tokens,
                    device=device,
                    gamma=gamma,
                    use_cache=use_cache,
                    profiler=profiler
                )
                
                total_times.append(total_ms)
                acceptance_rates.append(acceptance_rate)
                mean_accepted_list.append(mean_accepted)
                
                for stage, stats in breakdown.items():
                    if stage not in stage_accumulators:
                        stage_accumulators[stage] = []
                    stage_accumulators[stage].append(stats['total_ms'])
            
            num_prompts = len(prompts)
            mean_total_ms = sum(total_times) / num_prompts
            mean_acceptance_rate = sum(acceptance_rates) / num_prompts
            mean_accepted_tokens = sum(mean_accepted_list) / num_prompts
            
            mean_breakdown = {}
            for stage, ms_list in stage_accumulators.items():
                avg_ms = sum(ms_list) / num_prompts
                pct = (avg_ms / mean_total_ms * 100) if mean_total_ms > 0 else 0.0
                mean_breakdown[stage] = {
                    'avg_ms': round(avg_ms, 2),
                    'pct': round(pct, 2)
                }
            
            results.append({
                'gamma': gamma,
                'use_cache': use_cache,
                'mean_total_ms': mean_total_ms,
                'mean_acceptance_rate': mean_acceptance_rate,
                'mean_accepted_tokens': mean_accepted_tokens,
                'draft_speculation_ms': mean_breakdown['draft_speculation']['avg_ms'],
                'draft_speculation_pct': mean_breakdown['draft_speculation']['pct'],
                'target_verification_ms': mean_breakdown['target_verification']['avg_ms'],
                'target_verification_pct': mean_breakdown['target_verification']['pct'],
                'comparison_logic_ms': mean_breakdown['comparison_logic']['avg_ms'],
                'comparison_logic_pct': mean_breakdown['comparison_logic']['pct'],
                'cache_rollback_ms': mean_breakdown['cache_rollback']['avg_ms'],
                'cache_rollback_pct': mean_breakdown['cache_rollback']['pct'],
                'tensor_updates_ms': mean_breakdown['tensor_updates']['avg_ms'],
                'tensor_updates_pct': mean_breakdown['tensor_updates']['pct']
            })
            
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path)
    print(f"\n>> Profiling results saved at {save_path}")

    
if __name__ == '__main__':
    print(f">> Profiling on: {DEVICE.upper()}\n")
    parser = argparse.ArgumentParser("Profiling different operations of speculative decoding")
    
    parser.add_argument(
        "--model", type=str, default="pythia",
        choices=["pythia", "SmolLM", "SmolLM2"], help="Choose model family to profile"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate"
    )
    args = parser.parse_args()
    
    model_family = args.model
    max_new_tokens = args.max_new_tokens
    
    if model_family == 'pythia':
        SAVE_PATH = os.path.join(RESULTS_DIR, "pythia_profile.csv")
        
        print(f">> Loading {model_family} models")
        main_model, main_tokenizer = get_model(model_name="pythia-1B")
        fix_pad(main_model, main_tokenizer)
        
        draft_model, draft_tokenizer = get_model(model_name="pythia-160M")
        fix_pad(draft_model, draft_tokenizer)
        
    elif model_family == 'SmolLM':
        SAVE_PATH = os.path.join(RESULTS_DIR, "smollm_profile.csv")
        
        print(f">> Loading {model_family} models")
        main_model, main_tokenizer = get_model(model_name="SmolLM-1.7B")
        fix_pad(main_model, main_tokenizer)
        
        draft_model, draft_tokenizer = get_model(model_name="SmolLM-135M")
        fix_pad(draft_model, draft_tokenizer)
        
    elif model_family == 'SmolLM2':
        SAVE_PATH = os.path.join(RESULTS_DIR, "smollm2_profile.csv")
        
        print(f">> Loading {model_family} models")
        main_model, main_tokenizer = get_model(model_name="SmolLM2-1.7B")
        fix_pad(main_model, main_tokenizer)
        
        draft_model, draft_tokenizer = get_model(model_name="SmolLM2-135M")
        fix_pad(draft_model, draft_tokenizer)
    
    else:
        print(f"\n>> Error: Unknown model setup. Supported models: pythia, SmolLM and SmolLM2")
        exit()
    
    if main_model and draft_model:
        main_model.eval()
        draft_model.eval()
        
        run_profiling_benchmark(
            main_model=main_model,
            draft_model=draft_model,
            tokenizer=main_tokenizer,
            save_path=SAVE_PATH,
            max_new_tokens=max_new_tokens
        )
        
    else:
        print(f"\n>> Error while loading models.")