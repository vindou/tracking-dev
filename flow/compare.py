"""Paired comparison of tracker results on the same scenes.   python -m flow.compare flow/data/eval2_a flow/data/eval2_b ...

The first folder is the reference. For every other folder: correct / wrong / none of both, the scenes that changed
(correct in one and not in the other), and the exact two-sided sign test on those (McNemar). Scenes are paired, so
this test is much sharper than comparing the two totals. Also the same for wrong acceptances."""
import sys
from math import comb
from pathlib import Path

import numpy as np

from .evaluate import read, verdict


def sign_test(a, b):
    """Two-sided exact binomial p-value for a wins against b wins."""
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(ev):
    ev = Path(ev)
    h = read(ev / "hints.csv", "hints")
    cat = [str(c) for c in np.load(ev / "meta.npz")["cat"]]
    # the hint path alone, as in the table of evaluate.py ("flow + ICP")
    return cat, [verdict(h[i]) if h[i]["by_hint"] == "1" else "none" for i in range(len(cat))]


def main():
    cat, ref = load(sys.argv[1])
    count = lambda v: " / ".join(str(sum(x == k for x in v)) for k in ("correct", "wrong", "none"))
    print(f"{Path(sys.argv[1]).name:28s} {count(ref)}   ({len(ref)} scenes; correct / wrong / none)")
    for other in sys.argv[2:]:
        c2, v = load(other)
        assert c2 == cat, "different scenes"
        gain = sum(a != "correct" and b == "correct" for a, b in zip(ref, v))
        loss = sum(a == "correct" and b != "correct" for a, b in zip(ref, v))
        wg = sum(a == "wrong" and b != "wrong" for a, b in zip(ref, v))
        wl = sum(a != "wrong" and b == "wrong" for a, b in zip(ref, v))
        print(f"{Path(other).name:28s} {count(v)}   correct: +{gain} -{loss} (p = {sign_test(gain, loss):.3f})   "
              f"wrong: -{wg} +{wl} (p = {sign_test(wg, wl):.3f})")
        per = []
        for k in dict.fromkeys(cat):
            i = [j for j, c in enumerate(cat) if c == k]
            per.append(f"{k} {sum(ref[j] == 'correct' for j in i)}->{sum(v[j] == 'correct' for j in i)}")
        print(" " * 29 + "correct by kind: " + ", ".join(per))


if __name__ == "__main__":
    main()
