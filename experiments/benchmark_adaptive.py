import argparse
import os
import time
from functools import partial

import pandas as pd
import torch

from experiments.benchmark_tps import fix_pad, calculate_tps
from experiments.runtime_stress_simulator import RuntimeStressSimulator
from inference.adaptive_controller import AdaptiveController
from inference.generate import get_model, generate as ar_generate
from inference.speculative_engine import generate_speculative_standard

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results/")
os.makedirs(RESULTS_DIR, exist_ok=True)

PROMPTS = [
    "The capital of France is",
    "Artificial Intelligence works by",
    "The history of the Roman Empire is vast and"
]

CONTEXT_LENGTHS = [32, 64, 128, 256, 512, 1024, 2048]

PHASES = [
    ("No Load (0-30%)", 0.00, 0.30, False),
    ("GPU Load (30-60%)", 0.30, 0.60, True),
    ("No Load (60-80%)", 0.60, 0.80, False),
    ("GPU Load (80-100%)", 0.80, 1.00, True),
]


def reset_both():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def phase_for_tokens(tokens_generated, context_length):
    fraction = tokens_generated / max(context_length, 1)
    for name, start, end, load in PHASES:
        if fraction < end or end == 1.0:
            return name, load
    return PHASES[-1][0], PHASES[-1][3]


def empty_phase_dict(default=0.0):
    return {name: default for name, _, _, _ in PHASES}


def run_vanilla_sd(main_model, draft_model, model_tokenizer,
                   context_length, gamma=5):
    """Run vanilla SD with fixed gamma, return overall TPS."""
    generate_func = partial(
        generate_speculative_standard,
        main_model,
        draft_model,
        gamma=gamma,
        use_cache=True
    )
    tps, _, _ = calculate_tps(
        generate_func=generate_func,
        max_new_tokens=context_length,
        use_cache=True,
        model_tokenizer=model_tokenizer,
        reset_callback=reset_both,
        prompts=PROMPTS
    )
    return tps


def run_ar_baseline(main_model, model_tokenizer, context_length):
    """Run AR baseline, return overall TPS."""
    tps, _, _ = calculate_tps(
        generate_func=main_model.generate,
        max_new_tokens=context_length,
        use_cache=True,
        model_tokenizer=model_tokenizer,
        reset_callback=reset_both,
        prompts=PROMPTS
    )
    return tps


def run_adaptive(main_model, draft_model, model_tokenizer,
                 context_length):
    """Run adaptive controller, return overall TPS (excludes init time)."""
    controller = AdaptiveController(
        alpha=0.2,
        eval_every=10,
        re_explore_every=None
    )
    generate_func = partial(
        generate_speculative_standard,
        main_model,
        draft_model
    )
    tps, _, _ = calculate_tps(
        generate_func=generate_func,
        max_new_tokens=context_length,
        use_cache=True,
        model_tokenizer=model_tokenizer,
        reset_callback=reset_both,
        prompts=PROMPTS,
        controller=controller,
        controller_main_model=main_model,
        controller_draft_model=draft_model,
        include_controller_init=False
    )
    return tps


def phase_token_counts(generated_tokens, context_length):
    """Return actual generated-token count falling into each configured phase."""
    counts = empty_phase_dict()
    for name, start, end, _ in PHASES:
        start_token = int(start * context_length)
        end_token = int(end * context_length)
        counts[name] = max(
            0,
            min(generated_tokens, end_token) - start_token
        )
    return counts


def run_dynamic_ar(main_model, model_tokenizer, context_length, simulator):
    """Run AR under the same changing-load schedule using generate.py."""
    total_tokens = 0
    total_time = 0.0
    total_phase_time = empty_phase_dict()
    total_phase_tokens = empty_phase_dict(0)

    simulator.prepare()
    simulator.set_active(False)

    for prompt_idx, prompt in enumerate(PROMPTS, 1):
        simulator.set_active(False)

        inputs = model_tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(DEVICE)
        attention_mask = inputs.attention_mask.to(DEVICE)
        reset_both()

        state = {
            "current_phase": PHASES[0][0],
            "phase_start": None,
            "phase_time": empty_phase_dict(),
        }

        def runtime_load_callback(tokens_generated):
            phase, load = phase_for_tokens(
                tokens_generated,
                context_length
            )

            now = time.perf_counter()

            if phase != state["current_phase"]:
                state["phase_time"][state["current_phase"]] += max(
                    0.0, now - state["phase_start"]
                )
                state["current_phase"] = phase
                state["phase_start"] = now
                simulator.set_active(load)

        simulator.set_active(False)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        state["phase_start"] = start

        with torch.no_grad():
            output_ids = ar_generate(
                model=main_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=model_tokenizer,
                max_new_tokens=context_length,
                use_cache=True,
                temperature=0.0,
                do_sample=False,
                runtime_load_callback=runtime_load_callback
            )

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed = end - start

        # Account for the final phase through the end of generation.
        state["phase_time"][state["current_phase"]] += max(
            0.0, end - state["phase_start"]
        )

        generated = output_ids.shape[1] - input_ids.shape[1]
        total_tokens += generated
        total_time += elapsed

        prompt_phase_tokens = phase_token_counts(
            generated, context_length
        )
        for phase in total_phase_tokens:
            total_phase_tokens[phase] += prompt_phase_tokens[phase]
            total_phase_time[phase] += state["phase_time"][phase]

        print(
            f"    AR Prompt {prompt_idx}: "
            f"{generated} tokens, {elapsed:.2f}s"
        )

    simulator.set_active(False)
    simulator.stop()

    overall_tps = total_tokens / max(total_time, 1e-9)
    phase_tps = {
        phase: (
            total_phase_tokens[phase]
            / max(total_phase_time[phase], 1e-9)
            if total_phase_tokens[phase] > 0 else 0.0
        )
        for phase in total_phase_tokens
    }

    return (
        overall_tps,
        phase_tps,
        total_phase_tokens,
        total_phase_time
    )


def run_dynamic_sd(main_model, draft_model, model_tokenizer,
                   context_length, simulator, adaptive=False, gamma=5):
    """Run SD under changing load and return overall + phase stats."""
    total_tokens = 0
    total_time = 0.0
    total_phase_tokens = empty_phase_dict(0)
    total_phase_time = empty_phase_dict()

    generate_func = partial(
        generate_speculative_standard,
        main_model,
        draft_model,
        gamma=gamma,
        use_cache=True
    )

    simulator.prepare()
    simulator.set_active(False)

    for prompt_idx, prompt in enumerate(PROMPTS, 1):
        simulator.set_active(False)
        inputs = model_tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(DEVICE)
        attention_mask = inputs.attention_mask.to(DEVICE)
        reset_both()

        active_controller = None
        if adaptive:
            active_controller = AdaptiveController(
                alpha=0.2,
                eval_every=10,
                re_explore_every=None
            )
            active_controller.initialize(
                main_model=main_model,
                draft_model=draft_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                device=DEVICE,
                use_cache=True
            )

        def step_callback(tokens_generated):
            phase, load = phase_for_tokens(tokens_generated, context_length)
            simulator.set_active(load)
            return phase

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output_ids, stats = generate_func(
                input_ids,
                max_new_tokens=context_length,
                use_cache=True,
                attention_mask=attention_mask,
                controller=active_controller,
                step_callback=step_callback,
                return_stats=True
            )

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        generated = output_ids.shape[1] - input_ids.shape[1]
        total_tokens += generated
        total_time += elapsed

        for phase in total_phase_tokens:
            total_phase_tokens[phase] += stats.get("phase_tokens", {}).get(phase, 0)
            phase_tps = stats.get("phase_tps", {}).get(phase, 0.0)
            if phase_tps > 0:
                phase_tokens = stats.get("phase_tokens", {}).get(phase, 0)
                total_phase_time[phase] += phase_tokens / phase_tps

        print(f"    {'Adaptive' if adaptive else 'Vanilla'} Prompt {prompt_idx}: "
              f"{generated} tokens, {elapsed:.2f}s")

    simulator.set_active(False)
    simulator.stop()

    overall_tps = total_tokens / max(total_time, 1e-9)
    phase_tps = {
        phase: total_phase_tokens[phase] / max(total_phase_time[phase], 1e-9)
        if total_phase_tokens[phase] > 0 else 0.0
        for phase in total_phase_tokens
    }
    return overall_tps, phase_tps, total_phase_tokens, total_phase_time


if __name__ == '__main__':
    print(f">> Benchmarking on {DEVICE.upper()}\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pythia",
                        choices=["pythia", "SmolLM", "SmolLM2"])
    parser.add_argument("--stress_fraction", type=float, default=0.3,
                        help="Fraction of GPU memory to consume in stress test (0.0-0.8)")
    args = parser.parse_args()

    model_name = args.model
    stress_frac = args.stress_fraction

    if model_name == "pythia":
        OVERALL_SAVE_PATH = os.path.join(RESULTS_DIR, "adaptive_overall_pythia.csv")
        LOADWISE_SAVE_PATH = os.path.join(RESULTS_DIR, "adaptive_loadwise_pythia.csv")
        main_model, model_tokenizer = get_model(model_name="pythia-1B")
        fix_pad(main_model, model_tokenizer)
        draft_model, _ = get_model(model_name="pythia-160M")
        fix_pad(draft_model, _)

    elif model_name == "SmolLM":
        OVERALL_SAVE_PATH = os.path.join(RESULTS_DIR, "adaptive_overall_smol.csv")
        LOADWISE_SAVE_PATH = os.path.join(RESULTS_DIR, "adaptive_loadwise_smol.csv")
        main_model, model_tokenizer = get_model(model_name="SmolLM-1.7B")
        fix_pad(main_model, model_tokenizer)
        draft_model, _ = get_model(model_name="SmolLM-135M")
        fix_pad(draft_model, _)

    elif model_name == "SmolLM2":
        OVERALL_SAVE_PATH = os.path.join(RESULTS_DIR, "adaptive_overall_smol2.csv")
        LOADWISE_SAVE_PATH = os.path.join(RESULTS_DIR, "adaptive_loadwise_smol2.csv")
        main_model, model_tokenizer = get_model(model_name="SmolLM2-1.7B")
        fix_pad(main_model, model_tokenizer)
        draft_model, _ = get_model(model_name="SmolLM2-135M")
        fix_pad(draft_model, _)

    main_model.eval()
    draft_model.eval()

    results_overall = []
    results_loadwise = []

    for context_length in CONTEXT_LENGTHS:
        print(f"\n{'='*60}")
        print(f">> Context length: {context_length} tokens")
        print(f"{'='*60}")

        print("\n[Condition 1] No background load")

        ar_tps = run_ar_baseline(main_model, model_tokenizer, context_length)
        print(f"  AR Baseline:           {ar_tps:.2f} TPS")
        time.sleep(5)

        vanilla_tps = run_vanilla_sd(
            main_model, draft_model, model_tokenizer,
            context_length, gamma=5
        )
        print(f"  Vanilla SD (γ=5):      {vanilla_tps:.2f} TPS")
        time.sleep(5)

        adaptive_tps = run_adaptive(
            main_model, draft_model, model_tokenizer, context_length
        )
        print(f"  Adaptive Controller:   {adaptive_tps:.2f} TPS")
        time.sleep(5)

        results_overall.append({
            "context_length": context_length,
            "condition": "No Load",
            "stress_fraction": 0.0,
            "AR Baseline": ar_tps,
            "Vanilla SD (γ=5)": vanilla_tps,
            "Adaptive Controller": adaptive_tps,
        })

        print(f"\n[Condition 2] Runtime-changing GPU load (stress={stress_frac:.0%})")

        # AR baseline under changing runtime conditions.
        dynamic_ar_tps, ar_phase_tps, ar_phase_tokens, ar_phase_time = run_dynamic_ar(
            main_model, model_tokenizer, context_length, simulator=RuntimeStressSimulator(
                device=DEVICE, memory_fraction=stress_frac
            )
        )
        print(f"  AR Baseline:           {dynamic_ar_tps:.2f} TPS")
        for phase, tps in ar_phase_tps.items():
            print(f"    {phase}: {tps:.2f} TPS")
        time.sleep(5)

        # Vanilla SD under the exact same changing runtime schedule.
        dynamic_vanilla_tps, vanilla_phase_tps, vanilla_phase_tokens, vanilla_phase_time = run_dynamic_sd(
            main_model, draft_model, model_tokenizer, context_length,
            simulator=RuntimeStressSimulator(device=DEVICE, memory_fraction=stress_frac),
            adaptive=False,
            gamma=5
        )
        print(f"  Vanilla SD (γ=5):      {dynamic_vanilla_tps:.2f} TPS")
        for phase, tps in vanilla_phase_tps.items():
            print(f"    {phase}: {tps:.2f} TPS")
        time.sleep(5)

        # Adaptive controller under the exact same changing runtime schedule.
        dynamic_adaptive_tps, adaptive_phase_tps, adaptive_phase_tokens, adaptive_phase_time = run_dynamic_sd(
            main_model, draft_model, model_tokenizer, context_length,
            simulator=RuntimeStressSimulator(device=DEVICE, memory_fraction=stress_frac),
            adaptive=True
        )
        print(f"  Adaptive Controller:   {dynamic_adaptive_tps:.2f} TPS")
        for phase, tps in adaptive_phase_tps.items():
            print(f"    {phase}: {tps:.2f} TPS")
        time.sleep(5)

        # Overall CSV: ONLY overall TPS. No phase-level columns here.
        results_overall.append({
            "context_length": context_length,
            "condition": "Runtime Changing Load",
            "stress_fraction": stress_frac,
            "AR Baseline": dynamic_ar_tps,
            "Vanilla SD (γ=5)": dynamic_vanilla_tps,
            "Adaptive Controller": dynamic_adaptive_tps,
        })

        # Load-wise CSV: one row per method x phase.
        method_phase_data = [
            ("AR Baseline", ar_phase_tps, ar_phase_tokens, ar_phase_time),
            ("Vanilla SD (γ=5)", vanilla_phase_tps, vanilla_phase_tokens, vanilla_phase_time),
            ("Adaptive Controller", adaptive_phase_tps, adaptive_phase_tokens, adaptive_phase_time),
        ]

        for method, phase_tps_dict, phase_tokens_dict, phase_time_dict in method_phase_data:
            for name, start_pct, end_pct, load in PHASES:
                results_loadwise.append({
                    "context_length": context_length,
                    "stress_fraction": stress_frac,
                    "method": method,
                    "phase": name,
                    "load_active": load,
                    "phase_start_pct": start_pct * 100,
                    "phase_end_pct": end_pct * 100,
                    "tokens": phase_tokens_dict[name],
                    "time_s": phase_time_dict[name],
                    "TPS": phase_tps_dict[name],
                })

        # Save after every context length in case of crash.
        overall_df = pd.DataFrame(results_overall)
        loadwise_df = pd.DataFrame(results_loadwise)
        overall_df.to_csv(OVERALL_SAVE_PATH, index=False)
        loadwise_df.to_csv(LOADWISE_SAVE_PATH, index=False)

        print(f"\n>> Overall results saved so far to {OVERALL_SAVE_PATH}")
        print(f">> Load-wise results saved so far to {LOADWISE_SAVE_PATH}")

    # Final save.
    overall_df = pd.DataFrame(results_overall)
    loadwise_df = pd.DataFrame(results_loadwise)
    overall_df.to_csv(OVERALL_SAVE_PATH, index=False)
    loadwise_df.to_csv(LOADWISE_SAVE_PATH, index=False)

    print(f"\n>> Final overall results saved to {OVERALL_SAVE_PATH}")
    print(f">> Final load-wise results saved to {LOADWISE_SAVE_PATH}")

    print(f"\n{'='*60}")
    print("SUMMARY — OVERALL TPS")
    print(f"{'='*60}")
    print(overall_df.to_string(index=False))

    print(f"\n{'='*60}")
    print("SUMMARY — LOAD-WISE TPS")
    print(f"{'='*60}")
    print(loadwise_df.to_string(index=False))
