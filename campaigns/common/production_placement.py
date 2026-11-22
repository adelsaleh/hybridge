"""Placement contract for full campaigns; smoke jobs are an explicit exception.

The CPU campaign belongs on CPU nodes, not CPU-only allocations on Hopper nodes.
Keep this module free of numerical imports so placement is checked before workers.
"""
from __future__ import annotations

import os

HOST = dict(resource="cpu", partition="genoa", cpus_per_task=16, gpus=0)
DEVICE = dict(resource="gpu", partition="hopper", cpus_per_task=16, gpus=1, gres="gpu:h200_nvl:1")


def resources(kind):
    if kind not in ("host", "device"):
        raise ValueError(f"Unknown production campaign kind: {kind}")
    return dict(HOST if kind == "host" else DEVICE)


def require_placement(kind, environment=None):
    """Reject an incorrect allocation before starting any production worker.

    The CPU partition is genoa. Smoke continues using its own
    mixed GPU/CPU path and does not call this full-campaign guard.
    """
    env = os.environ if environment is None else environment
    requested = resources(kind)
    if not env.get("SLURM_JOB_ID"):
        raise RuntimeError("Full campaign execution requires a Slurm allocation")
    if env.get("SLURM_JOB_PARTITION") != requested["partition"]:
        raise RuntimeError(f"Full {kind} campaign must run in {requested['partition']}, "
                           f"got {env.get('SLURM_JOB_PARTITION')}")
    identifiers = ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")
    counts = ("SLURM_GPUS_ON_NODE", "SLURM_GPUS", "SLURM_GPUS_PER_NODE", "SLURM_GPUS_PER_TASK")
    has_gpu = any(env.get(name, "").strip() for name in identifiers) or any(
        env.get(name, "").strip() not in ("", "0") for name in counts)
    if kind == "host":
        for name in (*identifiers, *counts):
            if (env.get(name, "").strip() if name in identifiers else env.get(name, "").strip() not in ("", "0")):
                raise RuntimeError(f"Full host campaign must have no GPU allocation: {name}={env[name]}")
    elif not has_gpu:
        raise RuntimeError("Full device campaign requires an allocated GPU")
    return requested
