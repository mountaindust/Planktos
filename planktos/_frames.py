'''
Per-frame data for Swarm.plot and Swarm.plot_all.

Each frame draws one recorded agent state and, in 2D, a fluid backdrop behind
it. Agent state comes from the Swarm's own history. The backdrop comes from a
run archive when the Environment has recorded one, and from the velocity field
otherwise.

FrameSource settles that once, and the plotting code indexes it.

Author: Christopher Strickland

Email: cstric12@utk.edu
'''

import warnings
from pathlib import Path

import numpy as np

from . import fluid as _fluid
from .archive import RunArchive, _quiver_name

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"


class FrameSource:
    '''The states a plot may draw, and where each one's data comes from.

    States are indexed 0 to n_states-1 and are the Swarm's recorded history
    followed by the present, so the last state is ``Swarm.positions`` at
    ``Environment.time``.

    A fluid backdrop is served from the run archive the Environment recorded,
    and from the velocity field otherwise. Reading it from a dynamically loaded
    field re-reads the dataset, so that case warns; a backdrop a recording lacks
    is refused.

    Parameters
    ----------
    swarm : Swarm
        the swarm being plotted
    fluid : {'vort', 'quiver'}, optional
        the backdrop to be drawn, which decides what has to be available
    clip : float, optional
        symmetric vorticity limit set by the caller, used in place of the
        archive's global one

    Attributes
    ----------
    times : ndarray or None
        the time of each state, in order. None when the histories are out of
        step with each other and there is no reliable time base
    n_states : int
    run : RunArchive or None
        the Environment's archive, when it has one
    strides : tuple of int or None
        quiver downsampling factors, once :meth:`resolve_strides` has settled
        them
    quiver_scale : float or None
        the arrow scale to draw with
    vort_clip : float or None
        the symmetric vorticity limit to draw with, or None to let it grow
        frame by frame
    '''

    # Stored quiver dumps held at once. Two is what the linear interpolation
    #   asks for, and consecutive frames share a bracketing pair, so a monotone
    #   sweep reads each dump once.
    QUIVER_CACHE = 2

    def __init__(self, swarm, fluid=None, clip=None):
        self.swarm = swarm
        self.envir = swarm.envir
        self.flow = self.envir.flow

        self.run = _open_archive(self.envir)
        self.times = _live_times(swarm)
        self.n_states = (len(swarm.pos_history) + 1 if self.times is None
                         else len(self.times))

        self._quiver_cache = {}
        self._restream = False
        self._plan_fluid(fluid, clip)


    ####################   set-up   ####################

    def _plan_fluid(self, fluid, clip):
        '''Settle where the backdrop comes from and what the scales are.'''

        self.strides = None
        self.quiver_scale = None
        # A limit the caller set is used as given; the archive's global one
        #   fills in below when there is none.
        self.vort_clip = clip
        self._vorticity_from = None
        self._quiver_from = None

        # 2D only, which is also the rule record() applies to fluid=.
        if self.flow is None or fluid is None or len(self.envir.L) == 3:
            return

        meta = (self.run.meta.get('fluid') or {}) if self.run is not None else {}
        quantities = tuple(meta.get('quantities') or ())
        # A field held whole gives the backdrop directly, exactly and with no
        #   I/O; a windowed one supplies it from disk.
        resident = not self.flow.is_windowed

        if fluid == 'vort':
            if resident:
                self._vorticity_from = 'field'
            elif 'vort' in quantities and meta.get('vorticity') in ('source',
                                                                    'archive'):
                self._vorticity_from = 'dumps'
                self.flow.vorticity_path = (meta.get('vorticity_dir')
                                            or self.run.path / 'fluid')
            elif self.run is not None:
                raise ValueError(_missing('vort', quantities))
            else:
                self._vorticity_from = 'field'
                self._restream = True

        elif fluid == 'quiver':
            if resident:
                self._quiver_from = 'field'
            elif 'quiver' in quantities:
                self._quiver_from = 'dumps'
                # Fixed when recording started, since plot_all derives arrow
                #   density from a figure size that did not exist then.
                self.strides = tuple(int(s) for s in meta['quiver_strides'])
            elif self.run is not None:
                raise ValueError(_missing('quiver', quantities))
            else:
                self._quiver_from = 'field'
                self._restream = True

        self._check_stored_coverage()
        self._global_scales(clip)


    def _check_stored_coverage(self):
        '''Refuse when the frames reach past the dumps a recording covers.

        A recording stopped before the run did leaves per-dump files for part of
        the fluid series, and the frames beyond them have nothing to read.
        '''

        if 'dumps' not in (self._vorticity_from, self._quiver_from):
            return
        if self.times is None or len(self.times) == 0:
            return
        first, last = _dump_span(self.flow.flow_times,
                                 float(np.min(self.times)),
                                 float(np.max(self.times)))
        if self._vorticity_from == 'dumps':
            directory = Path(self.flow.vorticity_path)
            _require_dumps('vort', range(first, last + 1),
                           lambda i: directory / self.flow.vorticity_filename(i))
        if self._quiver_from == 'dumps':
            directory = self.run.path / 'fluid'
            _require_dumps('quiver', range(first, last + 1),
                           lambda i: directory / _quiver_name(i))


    def _global_scales(self, clip):
        '''Settle the scales the whole render shares.

        Both come from a recording's per-dump extrema where there is one. Those
        are fixed once a dump has been read, where ``FluidData.fmax`` covers all
        data seen so far and goes on growing with every later fluid access -- so
        two renders of one recorded run agree on the arrow scale only if it
        comes from the recording. Colour limits additionally grow with each
        frame drawn, which a limit fixed before the first frame settles.
        '''

        if self._quiver_from is not None:
            self.quiver_scale = float(np.linalg.norm(np.array(self.flow.fmax)))

        stats = None if self.run is None else self.run.dump_stats()
        if stats is None:
            return
        # Both are already reduced over the dumps the run saw. NaN means it saw
        #   none, so the archive has nothing to say about the scale and the
        #   fallback above stands.
        if self._quiver_from is not None:
            per_component = np.asarray(stats['vmax'], dtype=float)
            if not np.all(np.isnan(per_component)):
                self.quiver_scale = float(np.linalg.norm(per_component))
        if self._vorticity_from is not None and clip is None:
            absmax = stats.get('vort_absmax')
            if absmax is not None and np.isfinite(absmax).all():
                self.vort_clip = float(absmax)


    ####################   the states   ####################

    def capture_at(self, t):
        '''Index of the state nearest in time to ``t``. Ties go to the earlier.'''

        if self.times is None or len(self.times) == 0:
            return self.n_states - 1
        return int(np.argmin(np.abs(self.times - t)))


    def time(self, n):
        '''Time of state ``n``.'''

        if self.times is not None:
            return float(self.times[n])
        # No reliable time base, so read the history, which still holds one
        #   entry per state.
        hist = self.envir.time_history
        return float(hist[n] if n < len(hist) else self.envir.time)


    def positions(self, n):
        '''Agent positions at state ``n``, as a masked array.'''

        hist = self.swarm.pos_history
        return self.swarm.positions if n >= len(hist) else hist[n]


    def velocities(self, n):
        '''Agent velocities at state ``n``, as a masked array.'''

        hist = self.swarm.vel_history
        return self.swarm.velocities if n >= len(hist) else hist[n]


    def props(self, n):
        '''The props DataFrame in force at state ``n``.

        The current props where no property history is being kept.
        '''

        hist = self.swarm.props_history
        if hist is None or n >= len(hist):
            return self.swarm.props
        return hist[n]


    ####################   the fluid   ####################

    def vorticity(self, time):
        '''The vorticity field to draw at ``time``.

        Taken by time rather than by state index: the fluid's own time base is
        the dump cadence, and a plot of the present has no state index.
        '''

        if self.flow.flow_times is None:
            return self.flow.get_vorticity()
        if self._vorticity_from == 'dumps':
            # Blended with the weights the velocity itself uses, so this is the
            #   curl of the field in use.
            return self.flow.get_stored_vorticity(time)
        return self.flow.get_vorticity(time=time)


    def quiver(self, time):
        '''The downsampled velocity components to draw at ``time``.

        Returns
        -------
        tuple of two ndarrays, each indexed [x,y] on the strided grid
        '''

        if self._quiver_from == 'dumps':
            # Subsampling is linear, so blending strided slices gives the
            #   strided slice of the blend.
            arrows = _fluid._linear_blend(self.flow.flow_times, time,
                                          self._quiver_dump)
            return arrows[0], arrows[1]
        if self.flow.flow_times is None:
            field = self.flow
        else:
            field = self.flow(time)
        M, N = self.strides
        return field[0][::M, ::N], field[1][::M, ::N]


    def _quiver_dump(self, t_idx):
        '''One dump's stored arrows, from the read cache.'''

        t_idx = int(t_idx)
        if t_idx not in self._quiver_cache:
            self._quiver_cache[t_idx] = self.run.quiver(t_idx)
        arrows = self._quiver_cache[t_idx]
        # Evict whatever is furthest from what was just asked for, which on a
        #   monotone sweep is the trailing dump.
        while len(self._quiver_cache) > self.QUIVER_CACHE:
            del self._quiver_cache[max(self._quiver_cache,
                                       key=lambda k: abs(k - t_idx))]
        return arrows


    def resolve_strides(self, figure_strides):
        '''Settle the quiver downsampling factors for this figure.

        The figure's own choice, except where the arrows come off disk: the
        grid they were stored on was fixed when recording started, and a figure
        that wanted a noticeably denser one is told so.

        Called once at figure setup; :meth:`quiver` uses the result.
        '''

        if self.strides is None:
            self.strides = tuple(int(s) for s in figure_strides)
            return self.strides
        # 1.5x, since rounding an arrow count against a grid lands a stride off
        #   by one routinely.
        if any(stored > 1.5*wanted for stored, wanted
               in zip(self.strides, figure_strides)):
            warnings.warn(
                'this figure would draw arrows every {} grid points, but the '
                'archive stores them every {} -- the quiver grid is fixed when '
                'recording starts, since the figure size does not exist then. '
                'Re-record with a larger quiver_shape for a denser one, or '
                'plot a smaller figure.'.format(tuple(figure_strides),
                                                self.strides), UserWarning)
        return self.strides


    def warn_if_restreaming(self, frame_times):
        '''Warn when drawing these frames will re-read the fluid dataset.'''

        # Only an unrecorded run reaches this; with a recording, a backdrop it
        #   lacks is refused up front.

        if not self._restream:
            return
        lo, hi = float(np.min(frame_times)), float(np.max(frame_times))
        times = self.flow.flow_times
        first, last = _dump_span(times, lo, hi)
        spanned = last - first + 1
        window = int(self.flow.INUM) + 1
        if spanned <= window:
            return
        warnings.warn(
            'drawing the fluid over {:g} to {:g} will re-read about {} of this '
            'dataset\'s {} dumps, because it is being loaded dynamically and '
            'only {} are in memory at a time. Recording the run '
            '(envir.record(...)) writes what a plot needs as the run proceeds, '
            'and a plot then costs no fluid reads at '
            'all.'.format(lo, hi, spanned, len(times), window), UserWarning)



####################   module helpers   ####################

def _open_archive(envir):
    '''The archive this Environment recorded to, opened for reading, or None.

    Set by ``Environment.record`` and kept after recording stops, so a plot of a
    finished run reads what that run wrote. An archive describing a fluid that
    has since been replaced is reported and passed over.
    '''

    path = getattr(envir, '_archive_path', None)
    if path is None:
        return None
    try:
        run = RunArchive(path)
        run.check_against(envir)
    except (OSError, ValueError) as err:
        warnings.warn(
            'the run archive at {} cannot be used for this plot ({}), so the '
            'fluid will be read from the field instead.'.format(path, err),
            UserWarning)
        return None
    return run


def _live_times(swarm):
    '''Times of the states available to draw: the history, then the present.

    None where the histories are out of step with each other, e.g. after
    ``move(update_time=False)`` without a matching environmental advance, since
    there is then no time base to select frames against.
    '''

    envir = swarm.envir
    n_hist = len(swarm.pos_history)
    if envir.time is None:
        # A step raised partway through, so the present positions hold a step
        #   applied to only some agents. The histories are a consistent record
        #   up to that point.
        return np.asarray(envir.time_history[:n_hist], dtype=float)
    if len(envir.time_history) < n_hist:
        return None
    return np.concatenate((np.asarray(envir.time_history[:n_hist], dtype=float),
                           (float(envir.time),)))


def _dump_span(times, lo, hi):
    '''First and last dump index an interpolation over ``[lo, hi]`` reads.

    Clamped to the series, matching the constant extrapolation the field itself
    applies outside its time bounds.
    '''

    end = len(times) - 1
    first = np.clip(np.searchsorted(times, lo, side='right') - 1, 0, end)
    last = np.clip(np.searchsorted(times, hi, side='left'), 0, end)
    return int(first), int(last)


def _require_dumps(quantity, indices, path_of):
    '''Raise unless every dump in ``indices`` has a per-dump file.'''

    missing = [i for i in indices if not path_of(i).is_file()]
    if not missing:
        return
    span = ('{}'.format(missing[0]) if len(missing) == 1
            else '{}-{}'.format(missing[0], missing[-1]))
    raise ValueError(
        "this run's recording covers only part of the fluid series: the run "
        "spans dumps {}-{} and '{}' was recorded for all but {}, so the run "
        "carried on after the recording stopped. Record the whole run, load "
        "the fluid with INUM=None to keep all of it in memory, or plot with "
        "fluid=None.".format(list(indices)[0], list(indices)[-1], quantity,
                             span))


def _missing(quantity, quantities):
    '''The message for a fluid quantity the archive cannot supply.'''

    return (
        "this run's archive holds no '{}' data -- it was recorded with "
        "fluid={}, and the fluid is being loaded dynamically, so the field is "
        "no longer in memory. Re-record with fluid='{}' included, or load the "
        "fluid with INUM=None to keep all of it.".format(
            quantity, list(quantities) or None, quantity))
