"""Rank-zero progress output with the same ``-v 0/1/2`` scale as the runners.

Level 0 keeps final results, warnings, errors and interaction prompts. Level 1
adds setup/accepted-state summaries. Level 2 is the maximum and includes
SNES iterations, sensitivities, predictor guards, rejected trials and scalar
target iterations. Reporting never changes solver tolerances or acceptance.
"""
from __future__ import annotations
import time


class ProgressReporter:
    def __init__(self, comm, verbosity=2, *, started=None):
        if verbosity not in (0, 1, 2):
            raise ValueError("verbosity must be 0, 1 or 2")
        self.comm, self.verbosity = comm, verbosity
        self.started = time.perf_counter() if started is None else started
        self.pump = None

    def __call__(self, message, level=1):
        if self.comm.rank == 0:
            if level <= self.verbosity:
                print(f"[+{time.perf_counter()-self.started:9.3f}s] {message}", flush=True)
            # Rendering/event handling stays on rank zero and the main thread.
            # This callback performs no MPI communication or field evaluation.
            if self.pump is not None:
                self.pump()


def quiet_report(message, level=1):
    """Default for library use: the CLI opts in to progress reporting."""
