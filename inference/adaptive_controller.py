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
                   attention_mask, device, n_tokens=10, use_cache=True):
        """
        Measure draft/main TPS and AR Baseline, then builds a pruned exploration order.
        Call once before generation starts.

        Returns True if Speculative Decoding is worth attempting
        Returns False if draft is slower than main (SD is skipped entirely)
        """

        main_tps = self._quick_tps(main_model, input_ids,
                                   attention_mask, device, n_tokens, use_cache)
        draft_tps = self._quick_tps(draft_model, input_ids,
                                    attention_mask, device, n_tokens, use_cache)
        self.baseline_tps = main_tps
        self.draft_main_ratio = draft_tps / main_tps if main_tps > 0 else 0.0

        # seq_len = input_ids.shape[1]
        # main_ms = self._measure_cached_step_ms(
        #     main_model, seq_len, input_ids, device
        # )
        # draft_ms = self._measure_cached_step_ms(
        #     draft_model, seq_len, input_ids, device
        # )
        # self.draft_main_ratio = main_ms / draft_ms

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

    # def initialize(self, main_model, draft_model, input_ids,
    #                attention_mask, device, min_viable_ratio=1.5, use_cache=True):
    #     """
    #     Measure draft/main TPS and AR Baseline, then builds a pruned exploration order.
    #     Call once before generation starts.
    #
    #     Returns True if Speculative Decoding is worth attempting
    #     Returns False if draft is slower than main (SD is skipped entirely)
    #     """
    #
    #     seq_len = input_ids.shape[1]
    #
    #     short_len = seq_len
    #     medium_len = seq_len + 128
    #
    #     # Short seq ratio
    #     main_ms_short = self._measure_cached_step_ms(
    #         model=main_model, seq_len=short_len, input_ids=input_ids, device=device
    #     )
    #     draft_ms_short = self._measure_cached_step_ms(
    #         model=draft_model, seq_len=short_len, input_ids=input_ids, device=device
    #     )
    #     ratio_short = main_ms_short / draft_ms_short
    #
    #     # Medium seq ratio
    #     main_ms_medium = self._measure_cached_step_ms(
    #         model=main_model, seq_len=medium_len, input_ids=input_ids, device=device
    #     )
    #     draft_ms_medium = self._measure_cached_step_ms(
    #         model=draft_model, seq_len=medium_len, input_ids=input_ids, device=device
    #     )
    #     ratio_medium = main_ms_medium / draft_ms_medium
    #
    #     # Use the minimum - more conservative, prevents false positives
    #     self.draft_main_ratio = min(ratio_short, ratio_medium)
    #
    #     # Also track the trend - if ratio is declining, be more conservative
    #     ratio_trend = ratio_medium - ratio_short
    #
    #     print(f"[Controller] ratio_short={ratio_short:.2f}x  "
    #           f"ratio_medium={ratio_medium:.2f}x  "
    #           f"trend={ratio_trend:+.2f}  "
    #           f"using={self.draft_main_ratio:.2f}x")
    #
    #     # Adjust min_viable_ratio upwards if ratio is declining fast
    #     effective_min = min_viable_ratio
    #     if ratio_trend < -0.5:
    #         effective_min = min_viable_ratio * 1.3
    #         print(f"[Controller] declining ratio detected -> "
    #               f"raising threshold to {effective_min:.2f}x")
    #
    #     if self.draft_main_ratio < effective_min:
    #         print(f"[Controller] ratio {self.draft_main_ratio:.2f}x < "
    #               f"{effective_min:.2f}x -> AR fallback")
    #         self.fallen_back = True
    #         return False
    #
    #     if self.draft_main_ratio < 2.0:
    #         gamma_options = [3, 5, 7]
    #     else:
    #         gamma_options = [3, 5, 7, 10]
    #
    #     self.exploration_order = [(g, use_cache) for g in gamma_options]
    #
    #     default = (5, use_cache) if (5, use_cache) in self.exploration_order \
    #             else self.exploration_order[0]
    #     self.current_gamma, self.current_cache = default
    #     return True

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

        # # Only check no-regression AFTER exploration is complete
        # all_measured = all(v is not None for v in self.config_ema.values())
        #
        # if not all_measured:
        #     # Still exploring — just move to next config, no strike checks
        #     self._evaluate()
        #     return self.current_gamma, self.current_cache

        # # No-regression guard
        # if (self.baseline_tps is not None and
        #         self.current_ema < self.baseline_tps * 0.95):
        #     self.strike_count += 1
        #
        #     print(f"[Controller] step {self.step_count}, strike {self.strike_count}/{self.max_strikes} "
        #           f"(EMA {self.current_ema:.1f} < baseline {self.baseline_tps:.1f})")
        #     if self.strike_count >= self.max_strikes:
        #         print(f"[Controller] Falling back to AR Baseline")
        #         self.fallen_back = True
        #         return None, None
        # else:
        #     self.strike_count = 0

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

    @staticmethod
    def _quick_tps(model, input_ids, attention_mask, device, n_tokens=10, use_cache=True):
        """Run model for n_tokens, return TPS"""
        with torch.no_grad():
            start = time.time()
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=n_tokens,
                use_cache=use_cache
            )
            elapsed = time.time() - start
        return n_tokens/max(elapsed, 1e-9)

    @staticmethod
    def _measure_cached_step_ms(model, seq_len, input_ids, device):
        """
        Measure the cost of ONE cached generation step at a given sequence length.
        This is what actually happens during generation — not a full forward pass.
        """

        if input_ids.shape[1] < seq_len:
            pad = torch.ones(
                (input_ids.shape[0], seq_len - input_ids.shape[1]),
                dtype=input_ids.dtype, device=device
            )
            test_ids = torch.cat([input_ids, pad], dim=1)
        else:
            test_ids = input_ids[:, :seq_len]

        attn = torch.ones_like(test_ids)

        # Build cache by prefilling
        with torch.no_grad():
            out = model(test_ids, attention_mask=attn, use_cache=True)
            cache = out.past_key_values

        # Now measure cost of ONE new token step — this is the real cost
        new_token = torch.ones((input_ids.shape[0], 1),
                               dtype=input_ids.dtype, device=device)
        new_attn = torch.ones((input_ids.shape[0], seq_len + 1),
                              device=device)

        # Warmup
        with torch.no_grad():
            _ = model(new_token, attention_mask=new_attn,
                      past_key_values=cache, use_cache=True)

        # Measure
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(new_token, attention_mask=attn,
                      past_key_values=cache, use_cache=True)
        if device == 'cuda':
            torch.cuda.synchronize()

        return (time.time() - start) * 1000