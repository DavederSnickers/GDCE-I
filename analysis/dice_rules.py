"""Per-position edit rules: P(output != factual a) for a single categorical variable.
p = reverse posterior under the source class, q = under the target class, factual a = class 0
(the reference state). Replay at a state equal to the reference, so DICE's residual cancels exactly."""
import numpy as np
rng = np.random.default_rng(0)
N = 400_000
FLOOR = -np.log(1e-30)

def gumbel(shape):
    return -np.log(-np.log(rng.random(shape)))

def posterior_cf(p, q, a=0):
    """Gumbel-max counterfactual: abduct g | argmax(log p + g) = a (truncated Gumbel), replay under q"""
    L = np.log(p)[None, :].repeat(N, 0)
    Z = np.log(np.exp(np.log(p)).sum())
    top = Z + gumbel(N)
    g = L + gumbel(L.shape)
    val = -np.logaddexp(-g, -top[:, None])
    val[:, a] = top
    noise = val - L
    out = np.argmax(np.log(q)[None, :] + noise, 1)
    return (out != a).mean()

def dice(p, q, lam1, lam2, floor=FLOOR, a=0):
    onehot = np.full(len(p), -floor); onehot[a] = 0.0
    z = onehot - np.log(p)
    y = np.log(q)[None, :] + lam1 * z[None, :] + lam2 * gumbel((N, len(p)))
    return (np.argmax(y, 1) != a).mean()

def noinv(p, q, a=0):
    return 1 - q[a]

cases = [
    ('placebo, likely factual      p=q=(.9,.1)', [.9, .1], [.9, .1]),
    ('placebo, unlikely factual    p=q=(.1,.9)', [.1, .9], [.1, .9]),
    ('placebo, 3 classes unlikely  p=q=(.05,.9,.05)', [.05, .9, .05], [.05, .9, .05]),
    ('strong shift   p=(.9,.1)  q=(.1,.9)', [.9, .1], [.1, .9]),
    ('moderate shift p=(.9,.1)  q=(.6,.4)', [.9, .1], [.6, .4]),
    ('tiny shift, near-certain p=(.999,.001) q=(.99,.01)', [.999, .001], [.99, .01]),
    ('no-edge pos.   p=(.9999,.0001) q=(.999,.001)', [.9999, .0001], [.999, .001]),
]
rules = [('noinv', lambda p, q: noinv(p, q)),
         ('posterior', posterior_cf),
         ('DICE 5/95', lambda p, q: dice(p, q, .05, .95)),
         ('DICE 20/80', lambda p, q: dice(p, q, .2, .8)),
         ('DICE 100/0', lambda p, q: dice(p, q, 1, 0)),
         ('DICE 100/0 B=2', lambda p, q: dice(p, q, 1, 0, 2.0)),
         ('DICE 100/0 B=5', lambda p, q: dice(p, q, 1, 0, 5.0))]
print(f'{"P(edit)":52s}' + ''.join(f'{r:>15s}' for r, _ in rules))
for name, p, q in cases:
    p, q = np.array(p), np.array(q)
    print(f'{name:52s}' + ''.join(f'{f(p, q):15.4f}' for _, f in rules))
print('binary closed form for the posterior: max(0, 1 - q_a/p_a)')
for name, p, q in cases:
    if len(p) == 2:
        print(f'  {name:52s} {max(0, 1 - q[0] / p[0]):.4f}')
