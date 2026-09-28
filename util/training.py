"""
Helpers for sb-train.py's opponent mixes.
"""


def parse_edax_mix(depths, ratios):
    """'1,2,3', '0.3,0.3,0.3' -> ([1, 2, 3], [0.3, 0.3, 0.3]). Raises ValueError on bad input."""
    edax_depths = [int(d.strip()) for d in depths.split(',')]
    edax_ratios = [float(r.strip()) for r in ratios.split(',')]
    if len(edax_depths) != len(edax_ratios):
        raise ValueError(f"{len(edax_depths)} depths but {len(edax_ratios)} ratios")
    return edax_depths, edax_ratios


def mixed_block_plan(block_timesteps, selfplay_ratio, edax_depths, edax_ratios, random_ratio):
    """Split one training block among opponents, in the order they are trained.

    Ratios are normalized to sum to 1. Returns a list of (opponent, depth, timesteps) with
    opponent in {"Self", "Edax", "Random"} (depth is None except for Edax); zero-length
    entries are dropped. Rounding leftovers go to the last entry so the total is exact.
    """
    total = selfplay_ratio + sum(edax_ratios) + random_ratio
    if total <= 0:
        raise ValueError("ratios must sum to a positive number")
    entries = [("Self", None, selfplay_ratio)]
    entries += [("Edax", d, r) for d, r in zip(edax_depths, edax_ratios)]
    entries += [("Random", None, random_ratio)]
    entries = [(o, d, r / total) for o, d, r in entries if r > 0]
    plan = [(o, d, int(block_timesteps * r)) for o, d, r in entries]
    o, d, ts = plan[-1]
    plan[-1] = (o, d, ts + block_timesteps - sum(t for _, _, t in plan))
    return [(o, d, t) for o, d, t in plan if t > 0]
