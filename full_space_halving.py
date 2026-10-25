"""Exact full-space halving function exported from the frozen audit candidate.

Requires a Replay-compatible object. No model generation or GPU is used.
The Hyper argument is accepted for harness compatibility and is not used.
"""

def bl_successive_halving(replay, hp, budget, rng, **_):
    """Fixed-budget simple-regret baseline over every configuration.

    Rounds use common fresh instruction prefixes for all surviving arms.
    The remaining budget is allocated across the remaining halving rounds;
    every round is capped before querying, and ties use configuration IDs.
    No model generation, cross-target information, or response oracle is used.
    """
    import math
    replay.reset()
    alive = list(replay.configs)
    pool = list(replay.instructions)
    rng.shuffle(pool)
    cursor = 0
    hits = {c: 0 for c in alive}
    while len(alive) > 1 and replay.n_queries < budget:
        rounds_left = max(1, math.ceil(math.log2(len(alive))))
        remaining = budget - replay.n_queries
        planned = max(1, remaining // (len(alive) * rounds_left))
        width = min(planned, remaining // len(alive), len(pool) - cursor)
        if width <= 0:
            break
        fresh = pool[cursor:cursor + width]
        cursor += width
        for c in alive:
            hits[c] += sum(replay.query(c, x) for x in fresh)
        alive = sorted(alive, key=lambda c: (-hits[c], c))[:math.ceil(len(alive)/2)]
    return min(alive, key=lambda c: (-hits[c], c)), replay.n_queries
