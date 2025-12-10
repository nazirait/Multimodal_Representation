# src/comparative/utils/timer.py

import time

class Timer:
    def __init__(self, desc=None, verbose=True):
        self.desc = desc
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        if self.desc and self.verbose:
            print(f"[Timer] {self.desc} started...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.elapsed = self.end - self.start
        if self.verbose:
            print(f"[Timer] {self.desc or ''} finished in {self.elapsed:.2f} seconds.")

# Example usage:
# with Timer("Training epoch"):
#     ... code ...
