import argparse
import os
import random
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
    # General knowledge
    "The capital of France is",
    "The history of the Roman Empire is vast and",
    "One important reason cities develop near rivers is",

    # Science and nature
    "In a healthy ecosystem, plants and animals depend on each other because",
    "When scientists study climate change, they examine",
    "The process of energy transfer in a food chain begins when",

    # Computer science and technology
    "Artificial intelligence works by",
    "The process of sorting a list of numbers can be described as",
    "In machine learning, overfitting occurs when",

    # Explanatory and analytical writing
    "The main differences between renewable and nonrenewable energy sources are",
    "The debate about whether technology improves education often focuses on",
    "An efficient database system needs to balance",

    # Narrative writing
    "At dawn, the old railway station was almost empty. A woman carrying a blue suitcase",
    "The small research team opened the sealed container and discovered",
    "During the first week of the expedition, the weather changed so quickly that",
]

CONTEXT_LENGTHS = [64, 128, 256, 512, 1024]
GAMMAS = [2, 3, 5, 7, 10]
CSV_COLUMNS = (
    ['run_id', 'generation_length',
     'AR TPS', 'AR Tokens', 'AR Time']
    + [
        f"Vanilla SD (γ={gamma}) {metric}"
        for gamma in GAMMAS
        for metric in ['TPS', 'Tokens', 'Time']
    ]
    + ['Adaptive TPS', 'Adaptive Tokens', 'Adaptive Time']
)

def reset_both():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# def phase_for_tokens(tokens_generated, context_length):
#     fraction = tokens_generated / max(context_length, 1)
#     for name, start, end, load in PHASES:
#         if fraction < end or end == 1.0:
#             return name, load
#     return PHASES[-1][0], PHASES[-1][3]
#
#
# def empty_phase_dict(default=0.0):
#     return {name: default for name, _, _, _ in PHASES}


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
    tps, total_time, total_tokens = calculate_tps(
        generate_func=generate_func,
        max_new_tokens=context_length,
        use_cache=True,
        model_tokenizer=model_tokenizer,
        reset_callback=reset_both,
        prompts=PROMPTS
    )
    return tps, total_time, total_tokens


def run_ar_baseline(main_model, model_tokenizer, context_length):
    """Run AR baseline, return overall TPS."""
    tps, total_time, total_tokens = calculate_tps(
        generate_func=main_model.generate,
        max_new_tokens=context_length,
        use_cache=True,
        model_tokenizer=model_tokenizer,
        reset_callback=reset_both,
        prompts=PROMPTS
    )
    return tps, total_time, total_tokens


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
    tps, total_time, total_tokens = calculate_tps(
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
    return tps, total_time, total_tokens

#
# def phase_token_counts(generated_tokens, context_length):
#     """Return actual generated-token count falling into each configured phase."""
#     counts = empty_phase_dict()
#     for name, start, end, _ in PHASES:
#         start_token = int(start * context_length)
#         end_token = int(end * context_length)
#         counts[name] = max(
#             0,
#             min(generated_tokens, end_token) - start_token
#         )
#     return counts
#
#
# def run_dynamic_ar(main_model, model_tokenizer, context_length, simulator):
#     """Run AR under the same changing-load schedule using generate.py."""
#     total_tokens = 0
#     total_time = 0.0
#     total_phase_time = empty_phase_dict()
#     total_phase_tokens = empty_phase_dict(0)
#
#     simulator.prepare()
#     simulator.set_active(False)
#
#     for prompt_idx, prompt in enumerate(PROMPTS, 1):
#         simulator.set_active(False)
#
#         inputs = model_tokenizer(prompt, return_tensors="pt")
#         input_ids = inputs.input_ids.to(DEVICE)
#         attention_mask = inputs.attention_mask.to(DEVICE)
#         reset_both()
#
#         state = {
#             "current_phase": PHASES[0][0],
#             "phase_start": None,
#             "phase_time": empty_phase_dict(),
#         }
#
#         def runtime_load_callback(tokens_generated):
#             phase, load = phase_for_tokens(
#                 tokens_generated,
#                 context_length
#             )
#
#             now = time.perf_counter()
#
#             if phase != state["current_phase"]:
#                 state["phase_time"][state["current_phase"]] += max(
#                     0.0, now - state["phase_start"]
#                 )
#                 state["current_phase"] = phase
#                 state["phase_start"] = now
#                 simulator.set_active(load)
#
#         simulator.set_active(False)
#
#         if DEVICE == "cuda":
#             torch.cuda.synchronize()
#         start = time.perf_counter()
#         state["phase_start"] = start
#
#         with torch.no_grad():
#             output_ids = ar_generate(
#                 model=main_model,
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 tokenizer=model_tokenizer,
#                 max_new_tokens=context_length,
#                 use_cache=True,
#                 temperature=0.0,
#                 do_sample=False,
#                 runtime_load_callback=runtime_load_callback
#             )
#
#         if DEVICE == "cuda":
#             torch.cuda.synchronize()
#         end = time.perf_counter()
#         elapsed = end - start
#
#         # Account for the final phase through the end of generation.
#         state["phase_time"][state["current_phase"]] += max(
#             0.0, end - state["phase_start"]
#         )
#
#         generated = output_ids.shape[1] - input_ids.shape[1]
#         total_tokens += generated
#         total_time += elapsed
#
#         prompt_phase_tokens = phase_token_counts(
#             generated, context_length
#         )
#         for phase in total_phase_tokens:
#             total_phase_tokens[phase] += prompt_phase_tokens[phase]
#             total_phase_time[phase] += state["phase_time"][phase]
#
#         print(
#             f"    AR Prompt {prompt_idx}: "
#             f"{generated} tokens, {elapsed:.2f}s"
#         )
#
#     simulator.set_active(False)
#     simulator.stop()
#
#     overall_tps = total_tokens / max(total_time, 1e-9)
#     phase_tps = {
#         phase: (
#             total_phase_tokens[phase]
#             / max(total_phase_time[phase], 1e-9)
#             if total_phase_tokens[phase] > 0 else 0.0
#         )
#         for phase in total_phase_tokens
#     }
#
#     return (
#         overall_tps,
#         phase_tps,
#         total_phase_tokens,
#         total_phase_time
#     )
#
#
# def run_dynamic_sd(main_model, draft_model, model_tokenizer,
#                    context_length, simulator, adaptive=False, gamma=5):
#     """Run SD under changing load and return overall + phase stats."""
#     total_tokens = 0
#     total_time = 0.0
#     total_phase_tokens = empty_phase_dict(0)
#     total_phase_time = empty_phase_dict()
#
#     simulator.prepare()
#     simulator.set_active(False)
#
#     for prompt_idx, prompt in enumerate(PROMPTS, 1):
#         simulator.set_active(False)
#
#         inputs = model_tokenizer(prompt, return_tensors="pt")
#         input_ids = inputs.input_ids.to(DEVICE)
#         attn_mask = inputs.attention_mask.to(DEVICE)
#         reset_both()
#
#         active_controller = None
#         sd_viable = True
#
#         if adaptive:
#             active_controller = AdaptiveController(
#                 alpha=0.2,
#                 eval_every=10,
#                 re_explore_every=None
#             )
#             with torch.no_grad():
#                 sd_viable = active_controller.initialize(
#                     main_model=main_model,
#                     draft_model=draft_model,
#                     input_ids=input_ids,
#                     attention_mask=attn_mask,
#                     device=DEVICE,
#                     use_cache=True
#                 )
#
#         phase_state = {
#             "current": PHASES[0][0],
#             "step_start": None,
#             "phase_time": empty_phase_dict(0.0),
#             "phase_tokens": empty_phase_dict(0),
#             "last_tokens": 0,
#         }
#
#         def step_callback(tokens_generated):
#             now   = time.perf_counter()
#             phase, load = phase_for_tokens(tokens_generated, context_length)
#
#             # Accumulate time spent in previous phase since last callback
#             if phase_state["step_start"] is not None:
#                 phase_state["phase_time"][phase_state["current"]] += (
#                     now - phase_state["step_start"]
#                 )
#
#             # Accumulate tokens produced since last callback
#             delta = tokens_generated - phase_state["last_tokens"]
#             phase_state["phase_tokens"][phase_state["current"]] += delta
#             phase_state["last_tokens"] = tokens_generated
#
#             # Switch phase and toggle simulator if boundary crossed
#             if phase != phase_state["current"]:
#                 phase_state["current"] = phase
#                 simulator.set_active(load)
#
#             phase_state["step_start"] = now
#
#         if DEVICE == "cuda":
#             torch.cuda.synchronize()
#         gen_start = time.perf_counter()
#         phase_state["step_start"] = gen_start
#
#         with torch.no_grad():
#             if not sd_viable:
#                 # Controller decided SD not viable — same as calculate_tps
#                 # fallback: use plain AR generation
#                 output_ids = main_model.generate(
#                     input_ids,
#                     attention_mask=attn_mask,
#                     max_new_tokens=context_length,
#                     use_cache=True
#                 )
#             elif adaptive:
#                 # Adaptive SD — controller decides gamma per step
#                 output_ids, stats = generate_speculative_standard(
#                     main_model,
#                     draft_model,
#                     input_ids,
#                     attention_mask=attn_mask,
#                     max_new_tokens=context_length,
#                     use_cache=True,
#                     controller=active_controller,
#                     step_callback=step_callback,
#                     return_stats=True
#                 )
#             else:
#                 # Vanilla SD — fixed gamma, no controller
#                 output_ids, stats = generate_speculative_standard(
#                     main_model,
#                     draft_model,
#                     input_ids,
#                     attention_mask=attn_mask,
#                     max_new_tokens=context_length,
#                     use_cache=True,
#                     gamma=gamma,
#                     step_callback=step_callback,
#                     return_stats=True
#                 )
#
#         if DEVICE == "cuda":
#             torch.cuda.synchronize()
#         gen_end = time.perf_counter()
#
#         final_elapsed = gen_end - phase_state["step_start"]
#         phase_state["phase_time"][phase_state["current"]] += final_elapsed
#
#         generated = output_ids.shape[1] - input_ids.shape[1]
#         remaining_tokens = generated - phase_state["last_tokens"]
#         phase_state["phase_tokens"][phase_state["current"]] += remaining_tokens
#
#         total_tokens += generated
#         total_time += gen_end - gen_start
#
#         for phase in total_phase_tokens:
#             total_phase_tokens[phase] += phase_state["phase_tokens"][phase]
#             total_phase_time[phase] += phase_state["phase_time"][phase]
#
#         print(f"    {'Adaptive' if adaptive else 'Vanilla'} "
#               f"Prompt {prompt_idx}: {generated} tokens, "
#               f"{gen_end - gen_start:.2f}s")
#
#     simulator.set_active(False)
#     simulator.stop()
#
#     overall_tps = total_tokens / max(total_time, 1e-9)
#     phase_tps = {
#         phase: (total_phase_tokens[phase] /
#                 max(total_phase_time[phase], 1e-9))
#         if total_phase_tokens[phase] > 0 else 0.0
#         for phase in total_phase_tokens
#     }
#     return overall_tps, phase_tps, total_phase_tokens, total_phase_time


if __name__ == '__main__':
    print(f">> Benchmarking on {DEVICE.upper()}\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pythia",
                        choices=["pythia", "SmolLM", "SmolLM2"])
    # parser.add_argument("--stress_fraction", type=float, default=0.3,
    #                     help="Fraction of GPU memory to consume in stress test (0.0-0.8)")
    args = parser.parse_args()

    model_name = args.model
    # stress_frac = args.stress_fraction

    if model_name == "pythia":
        OVERALL_SAVE_PATH = os.path.join(RESULTS_DIR, "adaptive_overall_pythia.csv")
        # LOADWISE_SAVE_PATH = os.path.join(RESULTS_DIR, "adaptive_loadwise_pythia.csv")
        main_model, model_tokenizer = get_model(model_name="pythia-1B")
        fix_pad(main_model, model_tokenizer)
        draft_model, _ = get_model(model_name="pythia-160M")
        fix_pad(draft_model, _)

    elif model_name == "SmolLM":
        OVERALL_SAVE_PATH = os.path.join(RESULTS_DIR, "adaptive_overall_smol.csv")
        # LOADWISE_SAVE_PATH = os.path.join(RESULTS_DIR, "adaptive_loadwise_smol.csv")
        main_model, model_tokenizer = get_model(model_name="SmolLM-1.7B")
        fix_pad(main_model, model_tokenizer)
        draft_model, _ = get_model(model_name="SmolLM-135M")
        fix_pad(draft_model, _)

    elif model_name == "SmolLM2":
        OVERALL_SAVE_PATH = os.path.join(RESULTS_DIR, "adaptive_overall_smol2.csv")
        # LOADWISE_SAVE_PATH = os.path.join(RESULTS_DIR, "adaptive_loadwise_smol2.csv")
        main_model, model_tokenizer = get_model(model_name="SmolLM2-1.7B")
        fix_pad(main_model, model_tokenizer)
        draft_model, _ = get_model(model_name="SmolLM2-135M")
        fix_pad(draft_model, _)

    main_model.eval()
    draft_model.eval()

    # results_overall = []
    # results_loadwise = []

    if os.path.exists(OVERALL_SAVE_PATH):
        existing_df = pd.read_csv(OVERALL_SAVE_PATH)
        run_id = int(existing_df['run_id'].max()) + 1
    else:
        run_id = 1

    print(f">> Starting benchmark run: {run_id}")

    rng = random.Random(run_id)

    for context_length in CONTEXT_LENGTHS:
        print(f"\n{'='*60}")
        print(f">> Context length: {context_length} tokens")
        print(f"{'='*60}")

        methods = [
            ("ar", None),
            *[("vanilla", gamma) for gamma in GAMMAS],
            ("adaptive", None),
        ]
        rng.shuffle(methods)

        print(f">> Execution order: {methods}")

        row = {
            "run_id": run_id,
            "generation_length": context_length,
        }

        for method, gamma in methods:

            if method == "ar":
                ar_tps, ar_time, ar_tokens = run_ar_baseline(main_model, model_tokenizer, context_length)

                row.update({
                    "AR TPS": ar_tps,
                    "AR Tokens": ar_tokens,
                    "AR Time": ar_time,
                })

                print(
                    f"  AR Baseline:           {ar_tps:.2f} TPS | "
                    f"{ar_time:.2f} s | {ar_tokens} tokens"
                )

            elif method == "vanilla":
                vanilla_tps, vanilla_time, vanilla_tokens = run_vanilla_sd(
                    main_model, draft_model, model_tokenizer,
                    context_length, gamma=gamma
                )

                row.update({
                    f"Vanilla SD (γ={gamma}) TPS": vanilla_tps,
                    f"Vanilla SD (γ={gamma}) Tokens": vanilla_tokens,
                    f"Vanilla SD (γ={gamma}) Time": vanilla_time,
                })

                print(
                    f"  Vanilla SD (γ={gamma}): {vanilla_tps:.2f} TPS | "
                    f"{vanilla_time:.2f} s | {vanilla_tokens} tokens"
                )

            else:
                adaptive_tps, adaptive_time, adaptive_tokens = run_adaptive(
                    main_model, draft_model, model_tokenizer, context_length
                )

                row.update({
                    "Adaptive TPS": adaptive_tps,
                    "Adaptive Tokens": adaptive_tokens,
                    "Adaptive Time": adaptive_time,
                })
                print(
                    f"  Adaptive TPS:   {adaptive_tps:.2f} TPS | "
                    f"{adaptive_time:.2f} s | {adaptive_tokens} tokens"
                )

            time.sleep(5)

        # results_overall.append({
        #     "context_length": context_length,
        #     "AR TPS": ar_tps,
        #     "AR Tokens": ar_tokens,
        #     "AR Time": ar_time,
        #     **gamma_results,
        #     "Adaptive TPS": adaptive_tps,
        #     "Adaptive Tokens": adaptive_tokens,
        #     "Adaptive Time": adaptive_time,
        # })

        # print(f"\n[Condition 2] Runtime-changing GPU load (stress={stress_frac:.0%})")
        #
        # # AR baseline under changing runtime conditions.
        # dynamic_ar_tps, ar_phase_tps, ar_phase_tokens, ar_phase_time = run_dynamic_ar(
        #     main_model, model_tokenizer, context_length, simulator=RuntimeStressSimulator(
        #         device=DEVICE, memory_fraction=stress_frac
        #     )
        # )
        # print(f"  AR Baseline:           {dynamic_ar_tps:.2f} TPS")
        # for phase, tps in ar_phase_tps.items():
        #     print(f"    {phase}: {tps:.2f} TPS")
        # time.sleep(5)
        #
        # # Vanilla SD under the exact same changing runtime schedule.
        # dynamic_vanilla_tps, vanilla_phase_tps, vanilla_phase_tokens, vanilla_phase_time = run_dynamic_sd(
        #     main_model, draft_model, model_tokenizer, context_length,
        #     simulator=RuntimeStressSimulator(device=DEVICE, memory_fraction=stress_frac),
        #     adaptive=False,
        #     gamma=5
        # )
        # print(f"  Vanilla SD (γ=5):      {dynamic_vanilla_tps:.2f} TPS")
        # for phase, tps in vanilla_phase_tps.items():
        #     print(f"    {phase}: {tps:.2f} TPS")
        # time.sleep(5)
        #
        # # Adaptive controller under the exact same changing runtime schedule.
        # dynamic_adaptive_tps, adaptive_phase_tps, adaptive_phase_tokens, adaptive_phase_time = run_dynamic_sd(
        #     main_model, draft_model, model_tokenizer, context_length,
        #     simulator=RuntimeStressSimulator(device=DEVICE, memory_fraction=stress_frac),
        #     adaptive=True
        # )
        # print(f"  Adaptive Controller:   {dynamic_adaptive_tps:.2f} TPS")
        # for phase, tps in adaptive_phase_tps.items():
        #     print(f"    {phase}: {tps:.2f} TPS")
        # time.sleep(5)
        #
        # # Overall CSV: ONLY overall TPS. No phase-level columns here.
        # results_overall.append({
        #     "context_length": context_length,
        #     "condition": "Runtime Changing Load",
        #     "stress_fraction": stress_frac,
        #     "AR Baseline": dynamic_ar_tps,
        #     "Vanilla SD (γ=5)": dynamic_vanilla_tps,
        #     "Adaptive Controller": dynamic_adaptive_tps,
        # })
        #
        # # Load-wise CSV: one row per method x phase.
        # method_phase_data = [
        #     ("AR Baseline", ar_phase_tps, ar_phase_tokens, ar_phase_time),
        #     ("Vanilla SD (γ=5)", vanilla_phase_tps, vanilla_phase_tokens, vanilla_phase_time),
        #     ("Adaptive Controller", adaptive_phase_tps, adaptive_phase_tokens, adaptive_phase_time),
        # ]
        #
        # for method, phase_tps_dict, phase_tokens_dict, phase_time_dict in method_phase_data:
        #     for name, start_pct, end_pct, load in PHASES:
        #         results_loadwise.append({
        #             "context_length": context_length,
        #             "stress_fraction": stress_frac,
        #             "method": method,
        #             "phase": name,
        #             "load_active": load,
        #             "phase_start_pct": start_pct * 100,
        #             "phase_end_pct": end_pct * 100,
        #             "tokens": phase_tokens_dict[name],
        #             "time_s": phase_time_dict[name],
        #             "TPS": phase_tps_dict[name],
        #         })

        # Save after every context length in case of crash.
        # overall_df = pd.DataFrame(results_overall)
        # loadwise_df = pd.DataFrame(results_loadwise)
        # overall_df.to_csv(OVERALL_SAVE_PATH, index=False)
        # loadwise_df.to_csv(LOADWISE_SAVE_PATH, index=False)

        row_df = pd.DataFrame([row]).reindex(columns=CSV_COLUMNS)

        # Append to the master CSV; write its header only once.
        file_has_data = (
                os.path.exists(OVERALL_SAVE_PATH)
                and os.path.getsize(OVERALL_SAVE_PATH) > 0
        )

        row_df.to_csv(
            OVERALL_SAVE_PATH,
            mode="a",
            header=not file_has_data,
            index=False,
        )

        print(
            f">> Saved run {run_id}, generation length {context_length} "
            f"to {OVERALL_SAVE_PATH}"
        )

        # print(f"\n>> Overall results saved so far to {OVERALL_SAVE_PATH}")
        # print(f">> Load-wise results saved so far to {LOADWISE_SAVE_PATH}")

    # Final save.
    # overall_df = pd.DataFrame(results_overall)
    # loadwise_df = pd.DataFrame(results_loadwise)
    # overall_df.to_csv(OVERALL_SAVE_PATH, index=False)
    # loadwise_df.to_csv(LOADWISE_SAVE_PATH, index=False)

    # print(f"\n>> Final overall results saved to {OVERALL_SAVE_PATH}")
    # print(f">> Final load-wise results saved to {LOADWISE_SAVE_PATH}")

    # print(f"\n{'='*60}")
    # print("SUMMARY — LOAD-WISE TPS")
    # print(f"{'='*60}")
    # print(loadwise_df.to_string(index=False))
