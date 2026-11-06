"""Counterfactual marginal check: averaged over factuals a ~ Cat(p), does the replay under q
reproduce q (as any SCM counterfactual must)? Gumbel-max posterior vs DICE residual rules."""
import numpy as np
FLOOR = -np.log(1e-30)
def gumbel(shape):
    return -np.log(-np.log(rng.random(shape)))
rng = np.random.default_rng(1)
N = 300_000

def replay_posterior(p, q, a):
    L = np.log(p)[None, :].repeat(N, 0)
    top = np.log(p.sum()) + gumbel(N)
    g = L + gumbel(L.shape)
    val = -np.logaddexp(-g, -top[:, None]); val[:, a] = top
    return np.argmax(np.log(q)[None, :] + (val - L), 1)

def replay_dice(p, q, a, lam1, lam2, floor=FLOOR):
    oh = np.full(len(p), -floor); oh[a] = 0
    y = np.log(q)[None, :] + lam1 * (oh - np.log(p))[None, :] + lam2 * gumbel((N, len(p)))
    return np.argmax(y, 1)

for p, q in [([.9, .1], [.6, .4]), ([.5, .3, .2], [.2, .3, .5]), ([.7, .2, .1], [.7, .2, .1])]:
    p, q = np.array(p), np.array(q)
    print(f'p={p}  q={q}')
    for name, fn in [('posterior', replay_posterior),
                     ('DICE 5/95', lambda p, q, a: replay_dice(p, q, a, .05, .95)),
                     ('DICE 20/80', lambda p, q, a: replay_dice(p, q, a, .2, .8)),
                     ('DICE 100/0 B=2', lambda p, q, a: replay_dice(p, q, a, 1, 0, 2.0))]:
        marg = sum(p[a] * np.bincount(fn(p, q, a), minlength=len(p)) / N for a in range(len(p)))
        print(f'   {name:15s} counterfactual marginal {marg.round(3)}   (must equal q = {q})')
