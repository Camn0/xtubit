from __future__ import annotations
import torch


def solve_sb(Q, agents=256, max_steps=10000, mode="discrete", device="cuda"):
    try:
        import simulated_bifurcation as sb
    except ImportError as exc:
        raise RuntimeError("Install simulated-bifurcation before using the SB adapter") from exc
    result = sb.minimize(
        Q.to(device), domain="binary", agents=int(agents), best_only=False,
        max_steps=int(max_steps), mode=mode, early_stopping=False,
        device=device, verbose=False,
    )
    bits, values = result
    return bits, values
