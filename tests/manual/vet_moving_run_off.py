#! /usr/bin/env python3
'''Check the moving slider's run-off handling against an independent model.

Each trial is one agent step against one moving element (turning, stretching
and translating), drawn at random. The slider's end point
(_ibc.apply_internal_moving_BC) is compared with tests/_ib_harness.py's
moving_slide_model: a dense quadrature of the agent's signed distance along the
element, released at the first time it is off either end or the element moves
away from it, then free flight.

The table counts, per regime:
  stays on      the model keeps the agent in contact all step; the slider must
                end on the element, within tol of its final line
  end, transient / end, persistent
                the model releases the agent past an end; transient means it
                would have been back over the element by the end of the step
  rotate        the model releases the agent as the element moves away
  error         the slider raised
A trial counts as a match when the end points agree to tol, ten times the
slider's back-off distance _ibc._boundary_eps. The model does not follow the
agent after release, so a free flight that re-contacts the element shows up as a
mismatch; check any mismatch by hand before calling it a bug.

Run from the repository root:  python tests/manual/vet_moving_run_off.py
'''

import collections
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _ib_harness as h
from planktos import _geom, _ibc

warnings.simplefilter('ignore')

# name: (max turn per step in rad, element length range, max translation,
#        max agent step)
REGIMES = {
    'large turns, up to 1.5 rad': (1.5, (0.3, 2.0), 1.0, 2.5),
    'small rigid-ish turns, up to 0.1 rad': (0.1, (0.5, 1.5), 0.3, 1.0),
}
TRIALS = 6000
SEED = 1


def element(c, th, L):
    d = 0.5*L*np.array([np.cos(th), np.sin(th)])
    return np.array([[c - d, c + d]])


def classify(start, end, sm, em):
    try:
        got = h.call_moving(start, end, sm, em)[0]
    except Exception as err:
        return 'error', f'{type(err).__name__}: {str(err)[:60]}'
    model = h.moving_slide_model(start, end, sm, em)
    tol = 10*_ibc._boundary_eps(start, end, sm, em)
    if not model['released']:
        Q0, Q1 = em[0]
        L = np.linalg.norm(Q1 - Q0)
        s = np.dot(got - Q0, Q1 - Q0)/L**2
        on = abs(h.signed_perp_dist_2D(got, Q0, Q1)) < tol and \
            -tol/L <= s <= 1 + tol/L
        return 'stays on', 'match' if on else 'MISMATCH'
    kind = 'rotate' if model['why'] == 'rotate' else \
        'end, transient' if model['transient'] else 'end, persistent'
    d = np.linalg.norm(got - model['final'])
    return kind, 'match' if d < tol else f'MISMATCH ({d:.1e})'


def main():
    for name, (dmax, Lr, trm, stepm) in REGIMES.items():
        rng = np.random.default_rng(SEED)
        counts = collections.Counter()
        notes = []
        for trial in range(TRIALS):
            c = rng.uniform(4, 6, 2); th0 = rng.uniform(0, 2*np.pi)
            dth = rng.uniform(-dmax, dmax)
            L0, L1 = rng.uniform(*Lr, 2); tr = rng.uniform(-trm, trm, 2)
            sm = element(c, th0, L0); em = element(c + tr, th0 + dth, L1)
            start = c + rng.uniform(-1.5, 1.5, 2)
            end = start + rng.uniform(-stepm, stepm, 2)
            if _geom.seg_intersect_2D_multilinear_poly(
                    start, end, sm[:, 0], sm[:, 1], em[:, 0], em[:, 1]) is None:
                continue
            kind, verdict = classify(start, end, sm, em)
            counts[kind, verdict.split(' ')[0]] += 1
            if verdict != 'match':
                notes.append(f'  trial {trial}: {kind}, {verdict}')
        print(f'{name}  ({sum(counts.values())} contacts in {TRIALS} trials)')
        for kind in ('stays on', 'end, transient', 'end, persistent', 'rotate',
                     'error'):
            row = {v: n for (k, v), n in counts.items() if k == kind}
            if row:
                print(f'  {kind:16s}', ', '.join(f'{v} {n}' for v, n in
                                                sorted(row.items())))
        print('\n'.join(notes[:20]) if notes else '  no mismatches or errors')
        print()


if __name__ == '__main__':
    main()
