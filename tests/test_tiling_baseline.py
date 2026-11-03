'''Tiling off changes nothing: every untiled output, bitwise, against the branch point.

Scaffolding for the ``tiling`` branch (run_persistence.md 9.3), **deleted in the
commit that merges it back into ``dyload``** -- after that there is no "before" left
to compare against.

``_tiling_baseline_runner.py`` runs a set of untiled scenarios, one for each path
the tiling work edits -- analytic, IB2d (cubic, linear and windowed) and 3D vtk
fluid; static, moving and STL meshes; Brownian, tracer and inertial agents; FTLE;
recording, archive-served frames, restore and exports -- and saves the agent state
at every step, fluid accessor values inside and outside the domain, every archive
file, the exported vtk, and the rendered pixels of every plot. It is run here and
in a worktree at ``git merge-base HEAD dyload``, and the two outputs must agree
exactly.
'''

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
FIXTURES = TESTS / 'fixtures'
RUNNER = TESTS / '_tiling_baseline_runner.py'


def _git(*args):
    return subprocess.run(['git', *args], cwd=str(REPO), capture_output=True,
                          text=True)


def _run_side(root, work, out):
    script = Path(root) / '_tiling_baseline_check.py'
    shutil.copy(RUNNER, script)
    try:
        proc = subprocess.run(
            [sys.executable, str(script), str(TESTS), str(FIXTURES), str(work),
             str(out)],
            cwd=str(root), capture_output=True, text=True,
            env=dict(os.environ, MPLBACKEND='Agg'))
    finally:
        script.unlink(missing_ok=True)
    assert proc.returncode == 0, '{}:\n{}\n{}'.format(
        root, proc.stdout[-2000:], proc.stderr[-3000:])
    with np.load(out) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.slow
def test_an_untiled_run_is_bitwise_what_the_branch_point_produced(tmp_path):
    if shutil.which('git') is None:
        pytest.skip('git not on PATH')
    base = _git('merge-base', 'HEAD', 'dyload')
    if base.returncode != 0:
        pytest.skip('no dyload branch to find the branch point from')
    base = base.stdout.strip()

    worktree = tmp_path / 'base'
    made = _git('worktree', 'add', '--detach', str(worktree), base)
    if made.returncode != 0:
        pytest.skip('could not create a worktree: ' + made.stderr.strip())
    try:
        # Both sides use the same WORK, so paths an archive records agree.
        work = tmp_path / 'work'
        here = _run_side(REPO, work, tmp_path / 'here.npz')
        then = _run_side(worktree, work, tmp_path / 'base.npz')
    finally:
        _git('worktree', 'remove', '--force', str(worktree))
        _git('worktree', 'prune')

    assert sorted(here) == sorted(then), (
        'only here: {}; only at {}: {}'.format(
            sorted(set(here) - set(then)), base[:10],
            sorted(set(then) - set(here))))
    differ = []
    for key in sorted(here):
        a, b = here[key], then[key]
        same = (a.shape == b.shape and a.dtype == b.dtype and
                np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc'))
        if not same:
            differ.append(key)
    assert not differ, 'differs from {}: {}'.format(base[:10], differ)
