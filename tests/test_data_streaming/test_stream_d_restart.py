'''Claim 4 -- a recorded run can be picked up again from disk.

    "I should be able to run a simulation, streaming its results to disk, and
     then completely delete both the Environment and the Swarm object. I should
     then be able to create a new Environment and a new Swarm from that data on
     disk, recovering completely the state at which I left off the simulation so
     that I can resume it as if nothing happened."

This is **checkpoint and restart**, and ``docs/notes/run_persistence.md`` 2.11
names it as a follow-on that was deliberately not built: "Planktos has no
simulation checkpointing today, and adding it is the third problem this
architecture solves. Scope it separately; design the metadata for it now."
``TODO.md`` says the same about plotting a run in a later session.

So the tests here are not a discovery that something is broken. They are a
statement of what the claim needs, written as executable requirements, so that
the gap is measured rather than described:

* what the archive already carries, asserted positively -- the environment
  really can be rebuilt from provenance, and the agent state really does come
  back exactly;
* what it does not carry, one item per test, so the list is a checklist;
* and the whole claim end to end, attempted with everything today's public API
  offers, so that "as if nothing happened" is tested rather than assumed.

Every test marked xfail here is a piece of unbuilt work, not a defect in what
was built.
'''

import json
from pathlib import Path

import numpy as np
import pytest

import planktos

from _streaming import (assert_same_state, copy_ib2d, ib2d_envir, run)


# A goal line for work in progress rather than a regression suite: these run
# whole simulations and are opt-in, via --runstreaming. The members also marked
# slow (the example scripts, the cross-version check, the movie renders) need
# --runslow as well.
pytestmark = pytest.mark.streaming

DT = 0.5
STEPS = 8


def _record_a_run(tmp_path, steps=STEPS, seed=17, n=6):
    '''A recorded run, plus everything about it a restart would have to match.'''

    src = copy_ib2d(tmp_path, 'src', with_vorticity=True)
    envir = ib2d_envir(src, INUM=4)
    swrm = planktos.Swarm(swarm_size=n, envir=envir, seed=seed)
    swrm.shared_props['cov'] = swrm.shared_props['cov'] * 0.05
    swrm.add_prop('sensitivity', np.linspace(0.1, 0.9, n))
    with envir.record(str(tmp_path / 'run')) as rec:
        run(swrm, steps, dt=DT)
    return rec, envir, swrm


def _rebuild_environment(archive):
    '''An Environment built from nothing but the archive, as a user would.

    This is the recipe the provenance record exists to make possible: the
    loader's own name and the arguments it was called with, replayed.
    '''

    prov = archive.meta['provenance']
    env = prov['environment']
    L = env['L']
    kwargs = dict(Lx=L[0], Ly=L[1], units=env['units'],
                  rho=env['rho'], mu=env['mu'])
    if len(L) == 3:
        kwargs['Lz'] = L[2]
    bndry = env['bndry']
    for axis, name in zip(bndry, ('x_bndry', 'y_bndry', 'z_bndry')[:len(L)]):
        kwargs[name] = axis[0]
    envir = planktos.Environment(**kwargs)
    fluid = prov['fluid']
    getattr(envir, fluid['loader'])(**fluid['kwargs'])
    return envir


# --------------------------------------------------------------------------- #
#                       what the archive already carries                      #
# --------------------------------------------------------------------------- #

def test_the_environment_can_be_rebuilt_from_the_provenance_record(tmp_path):
    # The half that is built. Nothing about the fluid is serialized; the loader
    # call is replayed instead, and the archive's own fingerprint check is what
    # says the rebuilt world is the same world.
    rec, envir, swrm = _record_a_run(tmp_path)
    archive = planktos.load_run(rec.path)
    try:
        rebuilt = _rebuild_environment(archive)
        archive.check_against(rebuilt)             # raises if it is not
        assert list(rebuilt.L) == list(envir.L)
        np.testing.assert_array_equal(rebuilt.flow.flow_times,
                                      envir.flow.flow_times)
        for a, b in zip(rebuilt.flow.flow_points, envir.flow.flow_points):
            np.testing.assert_array_equal(a, b)
    finally:
        archive.close()


def test_the_agent_state_comes_back_exactly(tmp_path):
    # Positions, velocities and the clock survive the round trip bit for bit,
    # which is the part of a restart that the archive was actually built for.
    rec, envir, swrm = _record_a_run(tmp_path)
    last_pos = np.ma.copy(swrm.positions)
    last_vel = np.ma.copy(swrm.velocities)
    last_time = envir.time

    archive = planktos.load_run(rec.path)
    try:
        j = archive.capture_at(last_time)
        assert_same_state(archive.positions(0)[j], last_pos, 'positions')
        assert_same_state(archive.velocities(0)[j], last_vel, 'velocities')
        assert archive.times[j] == last_time
    finally:
        archive.close()


def test_a_hand_built_restart_gets_the_positions_and_the_clock_right(tmp_path):
    # As far as a knowledgeable user can get today with the public API. It is
    # worth pinning because it is the floor the missing pieces sit on top of.
    rec, envir, swrm = _record_a_run(tmp_path)
    want_pos = np.ma.copy(swrm.positions)
    want_time = envir.time
    del envir, swrm

    archive = planktos.load_run(rec.path)
    try:
        rebuilt = _rebuild_environment(archive)
        j = len(archive.times) - 1
        positions = np.ma.masked_invalid(
            np.ma.filled(archive.positions(0)[j], np.nan))
        resumed = planktos.Swarm(envir=rebuilt, init=np.ma.getdata(positions))
        resumed.positions = positions
        rebuilt.time = float(archive.times[j])
        rebuilt.time_history = [float(t) for t in archive.times[:j]]
    finally:
        archive.close()

    assert_same_state(resumed.positions, want_pos, 'restored positions')
    assert rebuilt.time == want_time


# --------------------------------------------------------------------------- #
#                     what the archive does not carry yet                     #
# --------------------------------------------------------------------------- #
# One item per test, from run_persistence.md 2.11's table of "what a checkpoint
# needs beyond the archive's last capture". Each is a piece of unbuilt work.

@pytest.mark.xfail(strict=True, reason=(
    "run_persistence.md 2.11: without Swarm.rndState a restart is not "
    "reproducible, 'which is most of the point'. The generator's state is a "
    "plain json-able dict and nothing writes it."))
def test_the_archive_carries_the_random_number_generator_state(tmp_path):
    rec, envir, swrm = _record_a_run(tmp_path)
    meta = json.loads((Path(rec.path) / 'meta.json').read_text())
    blob = json.dumps(meta)
    assert 'bit_generator' in blob or 'rndState' in blob


@pytest.mark.xfail(strict=True, reason=(
    "run_persistence.md 2.11: Swarm.props and shared_props are what make one "
    "agent differ from another. Nothing writes them, so a restarted swarm is "
    "a default swarm standing on recorded coordinates."))
def test_the_archive_carries_the_per_agent_properties(tmp_path):
    rec, envir, swrm = _record_a_run(tmp_path)
    blob = (Path(rec.path) / 'meta.json').read_text()
    assert 'sensitivity' in blob and 'cov' in blob


@pytest.mark.xfail(strict=True, reason=(
    "run_persistence.md 2.11: ib_condition, name and color are Swarm "
    "construction arguments a restart has to supply, and only 'name' is "
    "recorded (in the swarm sidecar)."))
def test_the_archive_carries_the_swarm_construction_arguments(tmp_path):
    rec, envir, swrm = _record_a_run(tmp_path)
    sidecar = json.loads(
        (Path(rec.path) / 'agents' / 'swarm00.json').read_text())
    blob = json.dumps(sidecar) + (Path(rec.path) / 'meta.json').read_text()
    for key in ('ib_condition', 'color'):
        assert key in blob, '{} is not recorded'.format(key)


@pytest.mark.xfail(strict=True, reason=(
    "There is no reader-side entry point for restarting: RunArchive exposes "
    "the numbers and the provenance, and nothing turns them back into an "
    "Environment and a Swarm. run_persistence.md 2.11 scopes this as a "
    "follow-on feature."))
def test_planktos_offers_a_way_to_resume_a_recorded_run(tmp_path):
    rec, envir, swrm = _record_a_run(tmp_path)
    archive = planktos.load_run(rec.path)
    try:
        entry_points = [name for name in
                        ('to_environment', 'restore', 'resume', 'rebuild',
                         'checkpoint')
                        if hasattr(archive, name)]
        entry_points += [name for name in
                         ('resume_run', 'restore_run', 'load_checkpoint')
                         if hasattr(planktos, name)]
        assert entry_points, 'no restart entry point exists'
    finally:
        archive.close()


# --------------------------------------------------------------------------- #
#                          the claim, end to end                              #
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason=(
    "Checkpoint and restart is not built (run_persistence.md 2.11). Even the "
    "most careful hand reconstruction diverges immediately, because the "
    "archive holds no RNG state, no props and no shared_props: the resumed "
    "swarm draws different random numbers from step one, and does so with "
    "default properties rather than the ones the run had."))
def test_a_run_resumes_from_disk_as_if_nothing_had_happened(tmp_path):
    # The reference: one uninterrupted run of 2*STEPS steps.
    src = copy_ib2d(tmp_path, 'ref_src', with_vorticity=True)
    ref_envir = ib2d_envir(src, INUM=4)
    ref = planktos.Swarm(swarm_size=6, envir=ref_envir, seed=17)
    ref.shared_props['cov'] = ref.shared_props['cov'] * 0.05
    ref.add_prop('sensitivity', np.linspace(0.1, 0.9, 6))
    run(ref, 2 * STEPS, dt=DT)

    # The claim: record the first half, throw everything away, rebuild, finish.
    rec, envir, swrm = _record_a_run(tmp_path, steps=STEPS)
    del envir, swrm

    archive = planktos.load_run(rec.path)
    try:
        rebuilt = _rebuild_environment(archive)
        j = len(archive.times) - 1
        positions = archive.positions(0)[j]
        resumed = planktos.Swarm(envir=rebuilt,
                                 init=np.ma.getdata(positions).copy())
        resumed.positions = np.ma.copy(positions)
        resumed.velocities = np.ma.copy(archive.velocities(0)[j])
        rebuilt.time = float(archive.times[j])
        rebuilt.time_history = [float(t) for t in archive.times[:j]]
    finally:
        archive.close()

    run(resumed, STEPS, dt=DT)

    assert rebuilt.time == pytest.approx(ref_envir.time)
    assert_same_state(resumed.positions, ref.positions,
                      'the resumed run ended somewhere else')
