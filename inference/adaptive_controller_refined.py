import statistics
import time

import torch


class AdaptiveControllerRefined:
    """Hardware-throughput-driven controller for speculative decoding.

    `update()` must be called after each speculative decoding step/chunk with
    the number of newly generated tokens and that chunk's elapsed time in ms.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor for per-configuration throughput measurements.
    eval_every : int
        Number of speculative steps/chunks in one measurement window.
    re_explore_every : int | None
        Number of exploitation steps/chunks between complete re-exploration
        rounds. None disables periodic re-exploration.
    min_improvement : float
        Required relative improvement to replace the incumbent after a fresh
        complete exploration round. 0.05 means 5%.
    verbose : int
        Print controller diagnostics when nonzero.
    min_fallback_ratio : float
        Keep the original early filter: if draft standalone TPS / target
        standalone TPS is below this ratio, recommend AR fallback.
    """

    def __init__(
        self,
        alpha=0.2,
        eval_every=10,
        re_explore_every=100,
        verbose=1,
        min_improvement=0.05,
        min_fallback_ratio=1.0,
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if eval_every < 1:
            raise ValueError("eval_every must be >= 1.")
        if re_explore_every is not None and re_explore_every < 1:
            raise ValueError("re_explore_every must be None or >= 1.")
        if min_improvement < 0:
            raise ValueError("min_improvement must be >= 0.")

        self.alpha = alpha
        self.eval_every = int(eval_every)
        self.re_explore_every = (
            None if re_explore_every is None else int(re_explore_every)
        )
        self.verbose = verbose
        self.min_improvement = float(min_improvement)
        self.min_fallback_ratio = float(min_fallback_ratio)

        self.exploration_order = []
        self.config_ema = {}       # Latest complete-round score per config
        self.round_scores = {}     # Fresh scores from the current round only
        self.baseline_tps = None
        self.draft_main_ratio = None

        self.current_gamma = 5
        self.current_cache = True
        self.step_count = 0         # All update() calls, including exploration
        self.window_step_count = 0  # update() calls in current measurement window
        self.current_ema = 0.0
        self.fallen_back = False

        self.phase = "explore"
        self.exploration_index = 0
        self.exploitation_steps = 0
        self.best_config = None
        self.best_tps = float("-inf")
        self._initialized = False

    def initialize(
        self,
        main_model,
        draft_model,
        input_ids,
        attention_mask,
        device,
        use_cache=True,
    ):
        """Calibrate standalone TPS, set candidates, and begin exploration.

        Returns False when the standalone draft/target ratio fails the early
        filter; otherwise returns True. This ratio is only a heuristic, not a
        guarantee that speculative decoding will beat AR end-to-end.
        """
        # Reset runtime state so the instance can be reused safely.
        self.config_ema = {}
        self.round_scores = {}
        self.step_count = 0
        self.window_step_count = 0
        self.current_ema = 0.0
        self.fallen_back = False
        self.phase = "explore"
        self.exploration_index = 0
        self.exploitation_steps = 0
        self.best_config = None
        self.best_tps = float("-inf")
        self._initialized = False

        main_tps = self._quick_tps(
            main_model, input_ids, attention_mask, device,
            warmup_tokens=20, measure_tokens=50, use_cache=use_cache,
        )
        draft_tps = self._quick_tps(
            draft_model, input_ids, attention_mask, device,
            warmup_tokens=20, measure_tokens=50, use_cache=use_cache,
        )

        self.baseline_tps = main_tps
        self.draft_main_ratio = draft_tps / main_tps if main_tps > 0 else 0.0

        if self.verbose:
            print(
                f"[Controller] main={main_tps:.1f} TPS "
                f"draft={draft_tps:.1f} TPS "
                f"ratio={self.draft_main_ratio:.2f}x"
            )

        if self.draft_main_ratio < self.min_fallback_ratio:
            if self.verbose:
                print(
                    f"[Controller] ratio < {self.min_fallback_ratio:.2f} "
                    "-> recommend AR fallback"
                )
            self.fallen_back = True
            self._initialized = True
            return False

        if self.draft_main_ratio < 2.0:
            gamma_options = [2, 3, 5, 7]
        else:
            gamma_options = [2, 3, 5, 7, 10]

        candidates = [(g, use_cache) for g in gamma_options]

        # Preserve the original preference for starting at gamma=5, while
        # still ensuring every candidate is visited exactly once per round.
        default = (5, use_cache) if (5, use_cache) in candidates else candidates[0]
        self.exploration_order = [default] + [cfg for cfg in candidates if cfg != default]
        self.config_ema = {cfg: None for cfg in self.exploration_order}
        self.round_scores = {}

        self.exploration_index = 0
        self.current_gamma, self.current_cache = default
        self.phase = "explore"
        self._initialized = True

        if self.verbose:
            print(
                f"[Controller] initial exploration: "
                f"γ={self.current_gamma} cache={self.current_cache}; "
                f"eval_every={self.eval_every}, "
                f"re_explore_every={self.re_explore_every}, "
                f"min_improvement={self.min_improvement:.1%}"
            )
        return True

    def update(self, tokens_generated, step_time_ms):
        """Record one speculative step/chunk and return config for the next.

        Returns (gamma, use_cache), or (None, None) when AR fallback was
        selected during initialization.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before update().")
        if self.fallen_back:
            return None, None
        if step_time_ms <= 0 or tokens_generated < 0:
            raise ValueError("step_time_ms must be > 0 and tokens_generated >= 0.")

        step_tps = tokens_generated / (step_time_ms / 1000.0)
        if self.window_step_count == 0:
            self.current_ema = step_tps
        else:
            self.current_ema = (
                self.alpha * step_tps + (1.0 - self.alpha) * self.current_ema
            )

        self.step_count += 1
        self.window_step_count += 1
        if self.phase == "exploit":
            self.exploitation_steps += 1

        # Evaluate only at the end of a complete measurement window.
        if self.window_step_count < self.eval_every:
            return self.current_gamma, self.current_cache

        self._evaluate_window()
        return self.current_gamma, self.current_cache

    def _evaluate_window(self):
        """Finish the current window and perform the phase-specific action."""
        cfg = (self.current_gamma, self.current_cache)
        measured_tps = self.current_ema
        self.window_step_count = 0
        self.current_ema = 0.0

        if self.phase == "explore":
            self.round_scores[cfg] = measured_tps
            self.config_ema[cfg] = measured_tps

            # Advance to the next candidate; the next update() calls measure it.
            self.exploration_index += 1
            if self.exploration_index < len(self.exploration_order):
                next_cfg = self.exploration_order[self.exploration_index]
                self.current_gamma, self.current_cache = next_cfg
                if self.verbose:
                    print(
                        f"[Controller] exploring γ={next_cfg[0]} "
                        f"cache={next_cfg[1]}"
                    )
                return

            # All candidates have fresh scores from this complete round.
            round_best = max(self.round_scores, key=self.round_scores.get)
            round_best_tps = self.round_scores[round_best]

            # First round: no incumbent exists, so take the measured winner.
            if self.best_config is None:
                chosen = round_best
                reason = "initial best"
            else:
                incumbent_tps = self.round_scores.get(self.best_config)
                if incumbent_tps is None:
                    # This should not occur because every configured candidate
                    # is measured in every round; guard against inconsistent state.
                    raise RuntimeError("Incumbent was not measured in this round.")
                improvement = (
                    (round_best_tps - incumbent_tps) / incumbent_tps
                    if incumbent_tps > 0 else float("inf")
                )
                if round_best != self.best_config and improvement >= self.min_improvement:
                    chosen = round_best
                    reason = f"challenger improved {improvement:.1%}"
                else:
                    chosen = self.best_config
                    reason = (
                        f"kept incumbent; challenger improvement "
                        f"{improvement:.1%} < {self.min_improvement:.1%}"
                        if round_best != self.best_config
                        else "incumbent remains best"
                    )

            self.best_config = chosen
            self.best_tps = self.round_scores[chosen]
            self.current_gamma, self.current_cache = chosen
            self.phase = "exploit"
            self.exploitation_steps = 0
            if self.verbose:
                print(
                    f"[Controller] entering exploitation: γ={chosen[0]} "
                    f"cache={chosen[1]} ({self.best_tps:.2f} TPS; {reason})"
                )
            self.round_scores = {}
            return

        # Exploitation: keep the incumbent; do not change it based on a single
        # window. Periodic exploration, if enabled, challenges it in a fresh round.
        if (
            self.re_explore_every is not None
            and self.exploitation_steps >= self.re_explore_every
        ):
            self.phase = "explore"
            self.exploration_index = 0
            self.round_scores = {}
            self.exploitation_steps = 0
            first_cfg = self.exploration_order[0]
            self.current_gamma, self.current_cache = first_cfg
            if self.verbose:
                print(
                    f"[Controller] starting a fresh exploration round "
                    f"after {self.re_explore_every} exploitation steps; "
                    f"testing γ={first_cfg[0]} first"
                )

    def _reset_exploration(self):
        """Compatibility helper: explicitly start a fresh exploration round."""
        if not self.exploration_order:
            return
        self.phase = "explore"
        self.exploration_index = 0
        self.round_scores = {}
        self.window_step_count = 0
        self.current_ema = 0.0
        self.exploitation_steps = 0
        self.current_gamma, self.current_cache = self.exploration_order[0]
        if self.verbose:
            print(
                f"[Controller] step {self.step_count}: re-exploring; "
                f"testing γ={self.current_gamma} first"
            )

    @staticmethod
    def _quick_tps(
        model,
        input_ids,
        attention_mask,
        device,
        warmup_tokens=20,
        measure_tokens=50,
        use_cache=True,
    ):
        """Estimate standalone generation TPS using median elapsed time."""
        device_obj = torch.device(device) if not isinstance(device, torch.device) else device

        with torch.inference_mode():
            warmup_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=warmup_tokens,
                use_cache=use_cache,
            )
            warmup_attn = torch.ones_like(warmup_out)

            elapsed_times = []
            token_counts = []
            for _ in range(3):
                if device_obj.type == "cuda":
                    torch.cuda.synchronize(device_obj)
                start = time.perf_counter()
                output = model.generate(
                    input_ids=warmup_out,
                    attention_mask=warmup_attn,
                    max_new_tokens=measure_tokens,
                    use_cache=use_cache,
                )
                if device_obj.type == "cuda":
                    torch.cuda.synchronize(device_obj)
                elapsed_times.append(time.perf_counter() - start)
                token_counts.append(output.shape[1] - warmup_out.shape[1])

        median_time = statistics.median(elapsed_times)
        median_tokens = statistics.median(token_counts)
        return median_tokens / max(median_time, 1e-9)
