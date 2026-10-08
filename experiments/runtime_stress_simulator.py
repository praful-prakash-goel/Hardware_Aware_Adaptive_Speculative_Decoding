import math
import torch
import threading
import time


class RuntimeStressSimulator:
    """
    Runs a background GPU workload.

    The stress tensors are allocated once. GPU load can be turned
    ON/OFF during generation without reallocating memory.

    memory_fraction is defined as the fraction of CURRENTLY FREE GPU
    memory that the COMPLETE stress workload (input + output) may use.
    """

    def __init__(
        self,
        device="cuda",
        memory_fraction=0.30,
        dtype=torch.float16,
        max_side=32768,
    ):
        self.device = torch.device(device)
        self.memory_fraction = memory_fraction
        self.dtype = dtype
        self.max_side = max_side

        self._active = False
        self._running = False
        self._thread = None

        self._stress_tensor = None
        self._stress_output = None
        self._stream = None

    def prepare(self):
        """Allocate the stress workload and start worker inactive."""
        if not torch.cuda.is_available():
            return

        if self._running:
            self._active = False
            return

        self._running = True
        self._active = False

        self._thread = threading.Thread(
            target=self._stress_loop,
            daemon=True,
        )
        self._thread.start()

        while self._stress_tensor is None and self._running:
            time.sleep(0.01)

    def start(self):
        """Start or activate the background GPU stress."""
        if not torch.cuda.is_available():
            print("[Simulator] CUDA not available, skipping stress")
            return

        if not self._running:
            self._running = True
            self._active = True

            self._thread = threading.Thread(
                target=self._stress_loop,
                daemon=True,
            )
            self._thread.start()

            while self._stress_tensor is None and self._running:
                time.sleep(0.01)

            print("[Simulator] Background GPU stress STARTED")
        else:
            self._active = True
            print("[Simulator] Background GPU stress ON")

    def set_active(self, active):
        """
        Turn GPU contention ON/OFF without reallocating memory.
        """
        if not self._running:
            if active:
                self.start()
            return

        if self._active != active:
            self._active = active
            print(
                f"[Simulator] Background GPU stress "
                f"{'ON' if active else 'OFF'}"
            )

    def stop(self):
        """Stop worker and release all stress tensors."""
        self._active = False
        self._running = False

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._thread = None

        if self._stream is not None:
            try:
                self._stream.synchronize()
            except Exception:
                pass
            self._stream = None

        self._stress_tensor = None
        self._stress_output = None

        torch.cuda.empty_cache()

        print("[Simulator] Background GPU stress STOPPED")

    def _stress_loop(self):
        try:
            torch.cuda.set_device(self.device)

            # Get memory AFTER the model has already been loaded.
            device_index = self.device.index if self.device.index is not None else 0
            free_mem, total_mem = torch.cuda.mem_get_info(device_index)

            bytes_per_element = torch.tensor(
                [],
                dtype=self.dtype,
            ).element_size()

            # Total memory budget for the WHOLE stress workload.
            # This includes both input and output of the matmul.
            stress_budget = int(
                free_mem * self.memory_fraction
            )

            # Input tensor and output tensor are approximately
            # the same size, so split the budget in half.
            matrix_budget = stress_budget // 2

            side = int(
                math.sqrt(
                    matrix_budget / bytes_per_element
                )
            )

            # Round down to a convenient multiple.
            side = (side // 256) * 256

            # Prevent ridiculously huge matrices.
            side = min(side, self.max_side)

            if side < 1024:
                raise RuntimeError(
                    f"Stress matrix too small: {side}"
                )

            self._stream = torch.cuda.Stream(device=device_index)

            # Allocate on the dedicated stress stream.
            with torch.cuda.stream(self._stream):
                self._stress_tensor = torch.randn(
                    side,
                    side,
                    device=self.device,
                    dtype=self.dtype,
                )

                # Preallocate output so every iteration does NOT
                # repeatedly allocate/free GPU memory.
                self._stress_output = torch.empty_like(
                    self._stress_tensor
                )

            self._stream.synchronize()

            actual_matrix_bytes = (
                side * side * bytes_per_element
            )

            actual_total_bytes = (
                2 * actual_matrix_bytes
            )

            print(
                "[Simulator] "
                f"GPU={self.device}, "
                f"free_before={free_mem / 1024**3:.2f} GB, "
                f"matrix={side}x{side}, "
                f"dtype={self.dtype}, "
                f"stress_memory="
                f"{actual_total_bytes / 1024**3:.2f} GB, "
                f"fraction="
                f"{actual_total_bytes / free_mem:.2%}"
            )

            while self._running:

                if self._active:

                    with torch.cuda.stream(self._stream):
                        torch.mm(
                            self._stress_tensor,
                            self._stress_tensor.T,
                            out=self._stress_output,
                        )

                    # Wait until the stress operation finishes.
                    self._stream.synchronize()

                else:
                    time.sleep(0.001)

        except Exception as e:
            print(
                f"[Simulator] stress loop error: {e}"
            )
            self._running = False