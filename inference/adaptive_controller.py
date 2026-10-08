import time
from bdb import effective

import torch

class AdaptiveController:
    def __init__(self,
                 alpha=0.2,
                 eval_every=10,
                 re_explore_every=None):
        """
        Args:
            alpha: EMA smoothing factor
            eval_every: evaluate and possibly switch every K steps
            re_explore_every: re-explore configurations every N steps
        """
        self.alpha = alpha
        self.eval_every = eval_every
        self.re_explore_every = re_explore_every

        # Set during initialization
        self.exploration_order = [] # List of (gamma, cache) combination to try
        self.config_ema = {} # (gamma, cache) -> ema_tps | None
        self.baseline_tps = None
        self.draft_main_ratio = None

        # Runtime state
        self.current_gamma = 5
        self.current_cache = True
        self.step_count = 0
        self.current_ema = 0.0
        self.fallen_back = False # True = Fall back to AR Baseline

    def initialize(self, main_model, draft_model, input_ids,
                   attention_mask, device, use_cache=True):
        """
        Measure draft/main TPS and AR Baseline, then builds a pruned exploration order.
        Call once before generation starts.

        Returns True if Speculative Decoding is worth attempting
        Returns False if draft is slower than main (SD is skipped entirely)
        """

        # main_tps = self._quick_tps(main_model, input_ids, attention_mask,
        #                            device, n_tokens, use_cache)
        # draft_tps = self._quick_tps(draft_model, input_ids, attention_mask,
        #                             device, n_tokens, use_cache)
        main_tps = self._quick_tps(main_model, input_ids,
                                   attention_mask, device,
                                   warmup_tokens=100,
                                   measure_tokens=100,
                                   use_cache=use_cache)
        draft_tps = self._quick_tps(draft_model, input_ids,
                                    attention_mask, device,
                                    warmup_tokens=100,
                                    measure_tokens=100,
                                    use_cache=use_cache)

        self.baseline_tps = main_tps
        self.draft_main_ratio = draft_tps / main_tps if main_tps > 0 else 0.0

        print(f"[Controller] main={main_tps:.1f} TPS "
              f"draft={draft_tps:.1f} TPS ratio={self.draft_main_ratio:.2f}x")

        # Prune based on ratio
        if self.draft_main_ratio < 1.0:
            # If draft is slower than main -> Fallback to AR Baseline
            print(f"[Controller] ratio < 1.0 -> falling back to AR Baseline")
            self.fallen_back = True
            return False
        elif self.draft_main_ratio < 2.0:
            gamma_options = [2, 3, 5, 7]
        else:
            gamma_options = [2, 3, 5, 7, 10]

        self.exploration_order = [(g, use_cache) for g in gamma_options]

        for cfg in self.exploration_order:
            self.config_ema[cfg] = None # None = Not yet explored

        default = (5, use_cache) if (5, use_cache) in self.exploration_order else self.exploration_order[0]
        self.current_gamma, self.current_cache = default
        return True

    def update(self, tokens_generated, step_time_ms):
        """
        Call after every speculative step

        Returns (gamma, use_cache):
            - Normal configuration to use for next chunk
            - (None, None) -> fall back to AR Baseline
        """
        if self.fallen_back:
            return None, None

        step_tps = tokens_generated / max(step_time_ms/1000, 1e-9)

        # Update EMA
        if self.current_ema == 0.0:
            self.current_ema = step_tps
        else:
            self.current_ema = (self.alpha * step_tps + (1 - self.alpha) * self.current_ema)

        self.step_count += 1

        # Only evaluate after every K steps; else early exit
        if self.step_count % self.eval_every != 0:
            return self.current_gamma, self.current_cache

        # Re-explore periodically
        if self.step_count > 0 and self.re_explore_every is not None:
            if self.step_count % self.re_explore_every == 0:
                self._reset_exploration()

        self._evaluate()

        return self.current_gamma, self.current_cache

    def _evaluate(self):
        """Store current EMA then decide: explore next or exploit best"""

        all_measured = all(v is not None for v in self.config_ema.values())

        if not all_measured:
            # Exploration phase, save ema only if None
            if self.config_ema[(self.current_gamma,
                                self.current_cache)] is None:
                self.config_ema[(self.current_gamma,
                                 self.current_cache)] = self.current_ema

            # Get all unexplored configs
            unmeasured = [cfg for cfg, v in self.config_ema.items()
                          if v is None]

            if unmeasured:
                for cfg in self.exploration_order:
                    if cfg in unmeasured:
                        next_cfg = cfg
                        break
                self.current_gamma, self.current_cache = next_cfg
                self.current_ema = 0.0
                print(f"[Controller] exploring γ={next_cfg[0]} "
                      f"cache={next_cfg[1]}")
                return

        # All measured -> exploit best
        self.config_ema[(self.current_gamma,
                         self.current_cache)] = self.current_ema

        best = max(self.config_ema, key=lambda k: self.config_ema[k])
        if (best[0] != self.current_gamma or
                best[1] != self.current_cache):
            print(f"[Controller] switching to best config "
                  f"γ={best[0]} cache={best[1]} "
                  f"({self.config_ema[best]:.1f} TPS)")
            self.current_gamma, self.current_cache = best
            self.current_ema = self.config_ema[best]

    def _reset_exploration(self):
        """Wipe stored EMAs so all configs get re-measured"""
        print(f"[Controller] step {self.step_count}: re-exploring "
              f"(sequence length changed)")
        for key in self.config_ema:
            self.config_ema[key] = None
        self.current_ema = 0.0

    # @staticmethod
    # def _quick_tps(model, input_ids, attention_mask, device, n_tokens=10, use_cache=True):
    #     """Run model for n_tokens, return TPS"""
    #     with torch.no_grad():
    #         start = time.time()
    #         model.generate(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             max_new_tokens=n_tokens,
    #             use_cache=use_cache
    #         )
    #         elapsed = time.time() - start
    #     return n_tokens/max(elapsed, 1e-9)

    @staticmethod
    def _quick_tps(model, input_ids, attention_mask, device,
                   warmup_tokens=100, measure_tokens=100, use_cache=True):
        """More tokens = more stable measurement, less noise."""
        with torch.no_grad():
            # Warmup
            warmup_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=warmup_tokens,
                use_cache=use_cache
            )
            warmup_attn = torch.ones_like(warmup_out)

            # Measure twice and average — reduces single-measurement noise
            times = []
            for _ in range(2):
                if device == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                model.generate(
                    input_ids=warmup_out,
                    attention_mask=warmup_attn,
                    max_new_tokens=measure_tokens,
                    use_cache=use_cache
                )
                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)

        # Use median of two measurements
        return measure_tokens / min(times)