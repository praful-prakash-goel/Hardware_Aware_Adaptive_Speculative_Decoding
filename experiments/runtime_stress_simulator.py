import torch
import threading
import time


class RuntimeStressSimulator:
    """
    Runs a background thread that consumes GPU compute/memory.
    The stress tensor is allocated once; load can be activated/deactivated
    during a generation without reallocating GPU memory.
    """

    def __init__(self, device='cuda', memory_fraction=0.3):
        self.device = device
        self.memory_fraction = memory_fraction
        self._active = False
        self._running = False
        self._thread = None
        self._stress_tensor = None

    def prepare(self):
        """Allocate the stress workload and start the worker inactive."""
        if not torch.cuda.is_available():
            return
        if self._running:
            self._active = False
            return
        self._running = True
        self._active = False
        self._thread = threading.Thread(
            target=self._stress_loop, daemon=True)
        self._thread.start()
        while self._stress_tensor is None and self._running:
            time.sleep(0.01)

    def start(self):
        """Allocate the stress workload and start the background worker active."""
        if not torch.cuda.is_available():
            print("[Simulator] CUDA not available, skipping stress")
            return

        if not self._running:
            self._running = True
            self._active = True
            self._thread = threading.Thread(
                target=self._stress_loop, daemon=True)
            self._thread.start()
            while self._stress_tensor is None and self._running:
                time.sleep(0.01)
            print("[Simulator] Background GPU stress STARTED")
        else:
            self._active = True
            print("[Simulator] Background GPU stress ON")

    def set_active(self, active):
        """Enable/disable GPU stress without reallocating the stress tensor."""
        if not self._running:
            if active:
                self.start()
            return
        if self._active != active:
            self._active = active
            print(f"[Simulator] Background GPU stress {'ON' if active else 'OFF'}")

    def stop(self):
        """Stop the worker and release the stress tensor."""
        self._active = False
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._stress_tensor is not None:
            del self._stress_tensor
            self._stress_tensor = None
            torch.cuda.empty_cache()
        print("[Simulator] Background GPU stress STOPPED")

    def _stress_loop(self):
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory
            target_elements = int(total_mem * self.memory_fraction / 4)
            side = min(int(target_elements ** 0.5), 8192)

            self._stress_tensor = torch.randn(
                side, side, device=self.device, dtype=torch.float32)

            while self._running:
                if self._active:
                    _ = torch.mm(self._stress_tensor, self._stress_tensor.T)
                    torch.cuda.synchronize()
                else:
                    time.sleep(0.001)

        except Exception as e:
            print(f"[Simulator] stress loop error: {e}")
            self._running = False