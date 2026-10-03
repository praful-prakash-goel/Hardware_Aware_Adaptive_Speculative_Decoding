from functools import partial
import torch
import time

from inference.adaptive_controller import AdaptiveController
from inference.speculative_engine import generate_speculative_standard


# noinspection unreachable-code
def generate_speculative(main_model, draft_model, input_ids, tokenizer,
                         attention_mask, max_new_tokens=512,
                         device='cuda', return_stats=False):
    """
    Drop-in replacement for generate_speculative_standard that adds
    an adaptive (gamma, use_cache) controller on top.

    The existing speculative engine is still unchanged - only the
    (gamma, use_cache) arguments fed to it change over time.
    """

    controller = AdaptiveController(
        alpha=0.2,
        eval_every=10,
        re_explore_every=None
    )

    sd_viable = controller.initialize(
        main_model=main_model,
        draft_model=draft_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        device=device,
        n_tokens=10
    )

    # If ratio < 1.0 -> use AR Baseline
    if not sd_viable:
        print(f"[generate_adaptive] SD not viable -> AR Baseline")
        output_ids = main_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
        if return_stats:
            return output_ids, {"mode": "Autoregressive", "reason": "ratio < 1.0"}

        return output_ids

    # Adaptive speculative generation
    generated_ids = input_ids.clone()
    current_attn = attention_mask.clone()
    tokens_generated = 0

    # Stats tracking
    steps = 0
    config_history = []
    tps_history = []

    current_gamma = controller.current_gamma
    current_cache = controller.current_cache

    while tokens_generated < max_new_tokens:
        remaining = max_new_tokens - tokens_generated

        if current_gamma is None:
            # AR fallback branch
            chunk_start = time.time()
            with torch.no_grad():
                outputs = main_model.generate(
                    generated_ids,
                    attention_mask=current_attn,
                    use_cache=current_cache if current_cache else False
                )
            next_token = torch.argmax(
                outputs.logits[:, -1, :], dim=-1, keepdim=True
            )
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_attn = torch.ones_like(generated_ids)
            tokens_this_step = 1
            step_ms = (time.time() - chunk_start) * 1000
        else:
            # SD branch
            chunk_tokens = min(current_gamma + 1, remaining)

            chunk_start = time.time()
            new_ids = generate_speculative_standard(
                main_model=main_model,
                draft_model=draft_model,
                input_ids=generated_ids,
                tokenizer=tokenizer,
                attention_mask=current_attn,
                max_new_tokens=chunk_tokens,
                gamma=current_gamma,
                use_cache=current_cache,
                return_stats=False
            )
            step_ms = (time.time() - chunk_start) * 1000

            tokens_this_step = new_ids.shape[1] - generated_ids.shape[1]
            generated_ids = new_ids
            current_attn = torch.ones_like(generated_ids)

        tokens_generated += tokens_this_step
        steps += 1
        config_history.append((current_gamma, current_cache))
        tps_history.append(tokens_this_step / max(step_ms / 1000, 1e-9))

        # Ask controller for next config
        current_gamma, current_cache = controller.update(
            tokens_generated=tokens_this_step,
            step_time_ms=step_ms
        )

    if return_stats:
        total_tps = tokens_generated / sum(
            t / max(v, 1e-9)
            for t, v in zip(
                [1] * len(tps_history), tps_history
            )
        ) if tps_history else 0.0

        # Count how many steps used each config
        from collections import Counter
        config_counts = Counter(config_history)

        stats = {
            "mode": "Adaptive",
            "tokens_generated": tokens_generated,
            "steps": steps,
            "config_counts": dict(config_counts),
            "fallen_back": controller.fallen_back,
            "baseline_tps": controller.baseline_tps,
            "total_tps": total_tps,
            "final_gamma": current_gamma,
            "final_cache": current_cache
        }