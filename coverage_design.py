#!/usr/bin/env python3
"""
Choose the screening subset by PAIR COVERAGE rather than count balance.

The previous objective minimised how unevenly each level and each pair appeared.
That produces a balanced design, but balance is not what the subset is for: the
22 cells exist to inform predictions about the 66 cells they do not probe, and
what governs that is which (framing, template) and (framing, order) combinations
the design has actually seen. The count-balanced design left libra_style paired
with json and latex_table but never data_dictionary -- which is gpt-5.1's
optimum, and it never reached the shortlist.

With 11 framings, 4 templates and 2 orders there are 44 framing-template pairs
and 22 framing-order pairs. Twenty-two cells can touch at most 22 of the former,
so the question is which half, and the answer should spread across templates
rather than concentrate.

Objective, in priority order:
  1. maximise distinct (framing, template) pairs covered
  2. maximise distinct (framing, order) pairs covered
  3. among ties, keep template and order counts as even as possible
"""
import numpy as np
from collections import Counter
from itertools import combinations

F = [f"F{i}" for i in range(11)]          # placeholders; real names substituted
T = [f"T{i}" for i in range(4)]
O = ["o0", "o1"]


def score(design):
    ft = {(c[0], c[1]) for c in design}
    fo = {(c[0], c[2]) for c in design}
    t = Counter(c[1] for c in design)
    o = Counter(c[2] for c in design)
    bal = np.std([t[x] for x in T]) + np.std([o[x] for x in O])
    return (len(ft), len(fo), -bal)


def search(F, T, O, n_per_framing=2, restarts=300000, seed=0):
    rng = np.random.default_rng(seed)
    opts = [(t, o) for t in T for o in O]
    best, best_s = None, (-1, -1, -1e9)
    for _ in range(restarts):
        d = []
        for f in F:
            for p in rng.choice(len(opts), size=n_per_framing, replace=False):
                d.append((f, opts[p][0], opts[p][1]))
        s = score(d)
        if s > best_s:
            best, best_s = d, s
    return best, best_s


if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, "/dpc/kuex0037/libra-artifact")
    from attacks.libra_framings import UTOS_TOP4, STAGE_B_FRAMINGS
    F = sorted(STAGE_B_FRAMINGS)
    T = sorted(UTOS_TOP4)
    O = ["libra_then_utos", "utos_then_libra"]

    for seed in range(3):
        d, s = search(F, T, O, seed=seed)
        print(f"seed {seed}: ft_pairs={s[0]}/44  fo_pairs={s[1]}/22  "
              f"balance_penalty={-s[2]:.4f}")

    d, s = search(F, T, O, seed=0)
    print("\ntemplates:", Counter(c[1] for c in d))
    print("orders   :", Counter(c[2] for c in d))
    print(f"\ndistinct framing-template pairs: {s[0]}/44")
    print(f"distinct framing-order pairs   : {s[1]}/22")
    print("\nSCREEN_DESIGN = [")
    for f, t, o in sorted(d):
        print(f'    "{f}|{t}|{o}",')
    print("]")
