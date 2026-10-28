"""
CASCADE spike inference on dff.npy, followed by quantification figures.

Runs in its own environment (CASCADE installed from git: python 3.7,
tensorflow 2.3, numpy 1.18), not the suite2p one. The suite2p env cannot run
CASCADE, and this env cannot read ops.npy (pickled with numpy 2), so the frame
rate comes from the config.

    conda activate Cascade
    python run_cascade_inference.py --config configs/my_experiment.yaml

Outputs per session, in <suite2p folder>/<model_name>/:
    spike_rate_prediction.npy   CASCADE output, expected spikes per frame (n_cells, n_frames), NaN at edges
    discrete_spikes.npy         object array, one array of spike frame indices per cell
    spike_counts.npy            discrete spikes binned per frame (n_cells, n_frames), int
    noise_level_distribution.png, noise_matched_ground_truth.png   CASCADE QC
    n_spikes_events_hist.png, dff_single_by_n_spikes.png, dff_avg_by_n_spikes.png,
    dff_linearity_isolated.png, example_cells_f_dff_rate_spikes.png   quantification
"""
import argparse
import os
import sys
import time
import warnings
# TensorFlow start-up messages (CUDA libraries not found, CPU instructions, XLA) are harmless on CPU:
# silence them. Must be set before tensorflow is imported (through cascade2p).
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import get_experimenter_analysis_folder, load_config


def import_cascade(cascade_repo):
    # cascade2p is used from the git clone, not pip-installed. Its plotting
    # helpers use paths relative to the repo (Pretrained_models, Ground_truth),
    # so the working directory is moved there, as in the CASCADE demos.
    sys.path.insert(0, cascade_repo)
    os.chdir(cascade_repo)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)  # numpy / tensorflow 2.3 deprecation notices
        import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from cascade2p import cascade
    from cascade2p.utils import plot_noise_level_distribution, plot_noise_matched_ground_truth
    from cascade2p.utils_discrete_spikes import infer_discrete_spikes
    return cascade, infer_discrete_spikes, plot_noise_level_distribution, plot_noise_matched_ground_truth


def find_suite2p_folders(experimenter, mice_ids, haas_path=False, suite2p_subpath=('suite2p', 'plane0'),
                         paths=None, experimenter_map=None):
    # Suite2p output folders of the given mice that have a dff.npy.
    analysis_root = get_experimenter_analysis_folder(experimenter, haas_path, paths, experimenter_map)
    suite2p_folders = []
    for mouse_id in mice_ids:
        mouse_folder = os.path.join(analysis_root, mouse_id)
        for session_id in sorted(os.listdir(mouse_folder)):
            suite2p_folder = os.path.join(mouse_folder, session_id, *suite2p_subpath)
            if os.path.exists(os.path.join(suite2p_folder, 'dff.npy')):
                suite2p_folders.append(os.path.abspath(suite2p_folder))
    return suite2p_folders


def spike_counts_per_frame(spike_frames, n_frames):
    # Discrete spike frame indices (one array per cell) -> spike count per frame.
    # No offset correction: CASCADE fixed its 1-frame offset upstream (issue #90, Oct 2026).
    # Spikes inferred with an older CASCADE are 1 frame early: rerun inference.
    counts = np.zeros((len(spike_frames), n_frames), dtype=int)
    for cell, frames in enumerate(spike_frames):
        frames = np.round(np.asarray(frames, dtype=float)).astype(int)
        frames = frames[(frames >= 0) & (frames < n_frames)]
        counts[cell] = np.bincount(frames, minlength=n_frames)
    return counts


def window_any(mask, before, after):
    # True at frame t if mask has any True frame in [t - before, t + after].
    csum = np.r_[0, np.cumsum(mask)]
    t = np.arange(mask.size)
    return csum[np.clip(t + after + 1, 0, mask.size)] - csum[np.clip(t - before, 0, mask.size)] > 0


def extract_snippets(traces, onsets_per_cell, pre, post, subtract_baseline=True):
    """Snippets around onsets (one array of onset frames per cell), baseline-subtracted on the pre-onset window
    unless subtract_baseline is False (e.g. for the inferred rate)."""
    snippets = np.array([traces[cell, onset - pre:onset + post]
                         for cell, onsets in enumerate(onsets_per_cell)
                         for onset in onsets
                         if onset - pre >= 0 and onset + post <= traces.shape[1]])
    if len(snippets) and pre > 0 and subtract_baseline:
        snippets = snippets - np.nanmean(snippets[:, :pre], axis=1, keepdims=True)
    return snippets


def finish_rate_axes(rate_axes, color='tab:blue'):
    # Same rate scale on every panel's second y-axis, legend on the first.
    if not rate_axes:
        return
    top = max(a.get_ylim()[1] for a in rate_axes)
    for a in rate_axes:
        a.set_ylim(0, top)
        a.tick_params(axis='y', colors=color, labelsize=7)
        a.set_ylabel('inferred rate (spk/s)', color=color, fontsize=7)
    rate_axes[0].legend(frameon=False, fontsize=8, loc='center right')


def save_figure(fig, save_folder, name):
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, f'{name}.png'), dpi=150)
    plt.close(fig)


def run_inference(dff, model_name, model_folder, frame_rate, save_folder, cascade, infer_discrete_spikes,
                  plot_noise_level_distribution, plot_noise_matched_ground_truth, overwrite, log, dff_noise=None):
    # CASCADE predictions and discrete spikes. Existing files are reused unless overwrite.
    # Predictions run on the full dff (CASCADE cannot take NaN). The noise level of each cell, which
    # selects the model ensemble and so calibrates the spike rate, is taken from dff_noise (dff_masked:
    # artifact frames ignored) when given.
    prob_path = os.path.join(save_folder, 'spike_rate_prediction.npy')
    spikes_path = os.path.join(save_folder, 'discrete_spikes.npy')

    if os.path.exists(prob_path) and not overwrite:
        log('Loading existing spike_rate_prediction.npy.')
        spike_prob = np.load(prob_path)
    else:
        noise_levels = plot_noise_level_distribution(
            dff if dff_noise is None else dff_noise, frame_rate,
            save_path=os.path.join(save_folder, 'noise_level_distribution.png'))
        plt.close('all')
        log(f'Median CASCADE noise level: {np.nanmedian(noise_levels):.2f}'
            + (' (artifact frames excluded).' if dff_noise is not None else '.'))
        plot_noise_matched_ground_truth(model_name, np.round(np.nanmedian(noise_levels)), frame_rate,
                                        nb_traces=8, duration=50,
                                        save_path=os.path.join(save_folder, 'noise_matched_ground_truth.png'))
        plt.close('all')
        log('Predicting spike rates.')
        spike_prob = cascade.predict(model_name, dff, model_folder=model_folder, trace_noise_levels=noise_levels)
        np.save(prob_path, spike_prob)

    if os.path.exists(spikes_path) and not overwrite:
        log('Loading existing discrete_spikes.npy.')
        spike_frames = list(np.load(spikes_path, allow_pickle=True))
    else:
        log('Inferring discrete spikes.')
        _, spike_frames = infer_discrete_spikes(spike_prob, model_name, model_folder=model_folder)
        # Ragged list -> object array, one array of spike frames per cell.
        spike_frames_arr = np.empty(len(spike_frames), dtype=object)
        for cell, frames in enumerate(spike_frames):
            spike_frames_arr[cell] = np.asarray(frames)
        np.save(spikes_path, spike_frames_arr)

    spikes = spike_counts_per_frame(spike_frames, dff.shape[1])
    np.save(os.path.join(save_folder, 'spike_counts.npy'), spikes)
    log(f'{spikes.sum()} discrete spikes, {spikes.sum() / spikes.shape[0] / (spikes.shape[1] / frame_rate):.2f} '
        f'spk/s per cell on average.')
    return spike_prob, spikes


def quantify(raw_traces, traces, rate, spikes, imaging_rate, save_folder, session_id, neuron_snr=None,
             artifacts=None, unreliable=None, excluded_cells=None, noise_rel=None, min_events_per_cell=3,
             detection_threshold_sd=3, log=print,
             gap_s=0.03, n_spikes_to_plot=(1, 2, 3, 5, 10, 20), t_before=0.2, t_after=1,
             clear_before=0.5, clear_after=0.5, n_single_examples=3, n_example_cells=5, example_snr_min=None,
             example_first_cells=True, example_t_start=180, example_t_length=60, seed=0):
    '''
    Quantification figures.
    raw_traces: F_raw (a.u.), traces: dF/F (%), rate: spk/s, spikes: discrete spike count per frame.
    artifacts: artifact frames (shaded in the example figure).
    unreliable: frames whose spikes are unreliable (on or near an artifact). Events
                whose dF/F or clear window contains such a frame are left out.
    excluded_cells: bool per cell, cells left out of every figure (e.g. collapsed baseline).
    noise_rel: per-cell frame noise (dF/F fraction, noise_rel.npy), for the per-cell peak / noise.
    min_events_per_cell: isolated events a cell needs for a given n to enter the per-cell figures and table.
    detection_threshold_sd: peak / noise_rel an isolated event needs; the smallest such spike count per cell
        is saved as min_detectable_spikes (csv column and .npy).
    Time is frames / imaging_rate, 0 at imaging start.
    '''
    rng = np.random.default_rng(seed)
    if excluded_cells is not None and excluded_cells.any():
        # No spike -> no event for excluded cells. Cell indices stay those of dff.
        spikes = np.where(excluded_cells[:, None], 0, spikes)
        session_id = f'{session_id} ({excluded_cells.sum()} cells excluded)'
    frame_timestamps = np.arange(traces.shape[1]) / imaging_rate
    pre, post = int(round(t_before * imaging_rate)), int(round(t_after * imaging_rate))
    cw_pre, cw_post = int(round(clear_before * imaging_rate)), int(round(clear_after * imaging_rate))
    # Onset frames usable for events: no unreliable frame in the dF/F or clear window around them.
    usable = (np.ones(traces.shape[1], dtype=bool) if unreliable is None
              else ~window_any(unreliable, max(pre, cw_pre), max(post, cw_post)))

    # Sort spiking event according to number of spike sum with rolling window of size n_frames = param * sampling rate
    # Spiking frames separated by <= gap_frames are merged into one event; event size = summed spikes
    gap_frames = max(1, int(round(gap_s * imaging_rate)))
    events_n_spikes = []
    events_onset = []  # onset frame of each event
    for cell_spikes in spikes:
        active = np.flatnonzero(cell_spikes > 0)
        if active.size == 0:
            events_n_spikes.append(np.array([], dtype=int))
            events_onset.append(np.array([], dtype=int))
            continue
        event_starts = np.r_[0, np.flatnonzero(np.diff(active) > gap_frames) + 1]
        n_spikes = np.round(np.add.reduceat(cell_spikes[active], event_starts)).astype(int)
        keep = (n_spikes > 0) & usable[active[event_starts]]
        events_n_spikes.append(n_spikes[keep])
        events_onset.append(active[event_starts][keep])

    # Per-cell histogram: occurrences of n-spike events (n >= 1)
    max_n = max(1, max((e.max() for e in events_n_spikes if e.size), default=1))
    cell_hists = np.array([np.bincount(e, minlength=max_n + 1)[1:] for e in events_n_spikes])
    totals = cell_hists.sum(axis=0)
    log(f'QC events (spiking frames < {gap_s} s apart merged, artifact windows left out): {totals.sum()} events | '
        + ', '.join(f'{n} spk: {totals[n - 1]}' for n in n_spikes_to_plot if n <= max_n)
        + f' | largest event {max_n} spikes')

    # Figure: histogram summed over all cells
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(np.arange(1, max_n + 1), cell_hists.sum(axis=0), color='k')
    ax.set_yscale('log')
    ax.set_xlabel(f'Number of spikes per event (gap {gap_s} s)')
    ax.set_ylabel('Occurrences (all cells)')
    ax.set_title(f'{session_id} - {cell_hists.shape[0]} cells')
    save_figure(fig, save_folder, 'n_spikes_events_hist')

    # Isolated frames: frame with n spikes and no other spiking frame within the clear window around it
    t = np.arange(-pre, post) / imaging_rate
    active_cumsum = np.concatenate([np.zeros((spikes.shape[0], 1), dtype=int), np.cumsum(spikes > 0, axis=1)], axis=1)
    frames = np.arange(spikes.shape[1])
    lo, hi = np.clip(frames - cw_pre, 0, None), np.clip(frames + cw_post + 1, None, spikes.shape[1])
    n_active_in_window = active_cumsum[:, hi] - active_cumsum[:, lo]  # (n_cells, n_frames), includes the frame itself
    isolated = (n_active_in_window == 1) & usable[None, :]
    iso_onsets = {n: [np.flatnonzero((spikes[cell] == n) & isolated[cell]) for cell in range(traces.shape[0])]
                  for n in n_spikes_to_plot}
    log(f'QC isolated frames (no other spike within -{clear_before}/+{clear_after} s): '
        + ', '.join(f'{n} spk: {sum(o.size for o in iso_onsets[n])}' for n in n_spikes_to_plot))

    # Single isolated responses: n_single_examples random events per n spikes, before averaging
    fig, axes = plt.subplots(len(n_spikes_to_plot), n_single_examples,
                             figsize=(3.4 * n_single_examples, 2.2 * len(n_spikes_to_plot)),
                             sharex=True, sharey='row', squeeze=False)
    single_rate_axes = []
    for row, n in enumerate(n_spikes_to_plot):
        # (cell, onset) of isolated n-spike frames with a full, NaN-free window
        candidates = [(cell, onset) for cell, onsets in enumerate(iso_onsets[n]) for onset in onsets
                      if onset - pre >= 0 and onset + post <= traces.shape[1]
                      and not np.isnan(traces[cell, onset - pre:onset + post]).any()]
        picks = rng.choice(len(candidates), size=min(n_single_examples, len(candidates)), replace=False) \
            if candidates else []
        snippets = extract_snippets(traces, iso_onsets[n], pre, post)
        for col in range(n_single_examples):
            ax = axes[row, col]
            ax.axvline(0, color='grey', ls='--', lw=0.8)
            ax.spines[['top', 'right']].set_visible(False)
            if col >= len(picks):
                ax.text(0.5, 0.5, 'no event', transform=ax.transAxes, ha='center', va='center', color='grey')
                continue
            cell, onset = candidates[picks[col]]
            single = traces[cell, onset - pre:onset + post]
            if pre > 0:
                single = single - np.nanmean(single[:pre])
            if len(snippets):
                ax.plot(t, np.nanmean(snippets, axis=0), color='tab:red', lw=1, alpha=0.6, label='mean isolated')
            ax.plot(t, single, color='k', lw=0.8, label='single event')
            # Discrete spikes in the window, as ticks at the bottom of the panel (height = count)
            window_counts = spikes[cell, onset - pre:onset + post]
            spk = window_counts > 0
            ax.vlines(t[spk], 0, 0.04 * window_counts[spk], color='tab:blue', lw=1.5,
                      transform=ax.get_xaxis_transform())
            # Inferred rate of this cell around this event, on a second y-axis.
            ax_r = ax.twinx()
            ax_r.plot(t, rate[cell, onset - pre:onset + post], color='tab:green', lw=1, label='inferred rate')
            single_rate_axes.append(ax_r)
            ax.set_title(f'cell {cell} - t = {onset / imaging_rate:.1f} s', fontsize=8)
        axes[row, 0].set_ylabel(f'{n} spike(s)\ndF/F % (bs)')
    finish_rate_axes(single_rate_axes, color='tab:green')
    for ax in axes[-1]:
        ax.set_xlabel('Time from spike (s)')
    axes[0, 0].legend(frameon=False, fontsize=7, loc='upper right')
    fig.suptitle(f'{session_id} - single isolated responses (spikes in blue, inferred rate in green)')
    save_figure(fig, save_folder, 'dff_single_by_n_spikes')

    # Dff average traces for single spike event, two-spike events, 3-spike events, 5, 10, 20 grid figure
    n_cols = int(np.ceil(np.sqrt(len(n_spikes_to_plot))))
    n_rows = int(np.ceil(len(n_spikes_to_plot) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(n_spikes_to_plot):]:
        ax.set_visible(False)

    iso_peaks = []  # (n, peak amplitude, sem at peak) for isolated frames
    rate_axes = []  # second y-axis per panel: mean inferred rate around isolated frames
    for ax, n in zip(axes.flat, n_spikes_to_plot):
        all_onsets = [events_onset[cell][events_n_spikes[cell] == n] for cell in range(traces.shape[0])]
        ax.axvline(0, color='grey', ls='--', lw=0.8)
        n_ev = []
        for onsets, color, label in [(all_onsets, 'k', 'all events'), (iso_onsets[n], 'tab:red', 'isolated frames')]:
            snippets = extract_snippets(traces, onsets, pre, post)
            n_ev.append(len(snippets))
            if len(snippets):
                mean, sem = np.nanmean(snippets, axis=0), np.nanstd(snippets, axis=0) / np.sqrt(len(snippets))
                ax.plot(t, mean, color=color, label=label)
                ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.3)
                if label == 'isolated frames':
                    peak_idx = pre + np.nanargmax(mean[pre:])  # peak of the average after onset
                    iso_peaks.append((n, mean[peak_idx], sem[peak_idx]))
                    rate_snips = extract_snippets(rate, onsets, pre, post, subtract_baseline=False)
                    ax_r = ax.twinx()
                    ax_r.plot(t, np.nanmean(rate_snips, axis=0), color='tab:blue', lw=1, label='inferred rate')
                    rate_axes.append(ax_r)
        ax.set_title(f'{n} spike(s) - {n_ev[0]} all / {n_ev[1]} isolated', fontsize=9)
    finish_rate_axes(rate_axes)
    axes.flat[0].legend(frameon=False, fontsize=8)
    for i, ax in enumerate(axes.flat[:len(n_spikes_to_plot)]):
        if i + n_cols >= len(n_spikes_to_plot):  # no visible panel below
            ax.set_xlabel('Time from event onset (s)')
            ax.xaxis.set_tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel('ΔdF/F % (baseline-subtracted)')
    fig.suptitle(session_id)
    save_figure(fig, save_folder, 'dff_avg_by_n_spikes')

    # Linearity: peak isolated ΔdF/F vs n spikes, linear fit through origin
    if iso_peaks:
        n_arr, amp, amp_sem = map(np.array, zip(*iso_peaks))
        slope = np.sum(n_arr * amp) / np.sum(n_arr ** 2)
        ss_tot = np.sum((amp - amp.mean()) ** 2)
        r2 = 1 - np.sum((amp - slope * n_arr) ** 2) / ss_tot if ss_tot > 0 else np.nan
        log('QC pooled isolated peaks: ' + ', '.join(f'{n} spk {a:.0f}%' for n, a in zip(n_arr, amp))
            + f' | fit through origin {slope:.1f} % / spike, R2 = {r2:.2f}')
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.errorbar(n_arr, amp, yerr=amp_sem, fmt='o', color='tab:red', capsize=3)
        x_fit = np.array([0, n_arr.max()])
        ax.plot(x_fit, slope * x_fit, color='k', ls='--', lw=1,
                label=f'{slope:.1f} % / spike, R² = {r2:.2f}')
        ax.set_xlim(0, n_arr.max() * 1.1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Spikes in isolated frame')
        ax.set_ylabel('Peak ΔdF/F (%)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(session_id, fontsize=9)
        save_figure(fig, save_folder, 'dff_linearity_isolated')

    # Per-cell responses to isolated n-spike frames: average per cell first (cells with >= min_events_per_cell
    # events), then statistics across cells, so cells with many events do not dominate.
    from scipy import stats
    import pandas as pd
    per_cell, rows = {}, []
    for n in n_spikes_to_plot:
        cells_n, means_n, rates_n = [], [], []
        for cell, onsets in enumerate(iso_onsets[n]):
            snip = extract_snippets(traces[cell:cell + 1], [onsets], pre, post)
            if len(snip) < min_events_per_cell:
                continue
            mean = np.nanmean(snip, axis=0)
            peak = float(np.nanmax(mean[pre:]))  # % dF/F, peak of the cell average after onset
            cells_n.append(cell)
            means_n.append(mean)
            rates_n.append(np.nanmean(extract_snippets(rate[cell:cell + 1], [onsets], pre, post,
                                                       subtract_baseline=False), axis=0))
            rows.append(dict(cell=cell, n_spikes=n, n_events=len(snip), peak_dff_pct=peak,
                             peak_over_noise=peak / 100 / noise_rel[cell] if noise_rel is not None else np.nan))
        per_cell[n] = (np.array(cells_n), np.array(means_n), np.array(rates_n))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3 * n_rows), sharex=True, sharey=True,
                             squeeze=False)
    for ax in axes.flat[len(n_spikes_to_plot):]:
        ax.set_visible(False)
    rate_axes = []
    for ax, n in zip(axes.flat, n_spikes_to_plot):
        cells_n, means_n, rates_n = per_cell[n]
        ax.axvline(0, color='grey', ls='--', lw=0.8)
        if len(cells_n):
            mean = np.nanmean(means_n, axis=0)
            ax.plot(t, mean, color='tab:red', label='ΔdF/F')
            if len(cells_n) > 1:
                ci = stats.t.ppf(0.975, len(cells_n) - 1) * np.nanstd(means_n, axis=0, ddof=1) / np.sqrt(len(cells_n))
                ax.fill_between(t, mean - ci, mean + ci, color='tab:red', alpha=0.3)
            ax_r = ax.twinx()
            ax_r.plot(t, np.nanmean(rates_n, axis=0), color='tab:blue', lw=1, label='inferred rate')
            rate_axes.append(ax_r)
        ax.set_title(f'{n} spike(s) - {len(cells_n)} cells', fontsize=9)
    finish_rate_axes(rate_axes)
    for i, ax in enumerate(axes.flat[:len(n_spikes_to_plot)]):
        if i + n_cols >= len(n_spikes_to_plot):
            ax.set_xlabel('Time from event onset (s)')
            ax.xaxis.set_tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel('ΔdF/F % (baseline-subtracted)')
    fig.suptitle(f'{session_id} - isolated frames, mean of cell averages ± 95% CI across cells '
                 f'(cells with >= {min_events_per_cell} events)', fontsize=9)
    save_figure(fig, save_folder, 'dff_avg_by_n_spikes_per_cell')

    if rows:
        df = pd.DataFrame(rows)
        # One row per cell, columns per n: n_events_<n>, peak_dff_pct_<n>, peak_over_noise_<n>.
        wide = df.pivot(index='cell', columns='n_spikes', values=['n_events', 'peak_dff_pct', 'peak_over_noise'])
        wide.columns = [f'{v}_{n}' for v, n in wide.columns]

        # QC: smallest measured spike count whose isolated-event peak reaches detection_threshold_sd x noise_rel.
        # NaN: excluded cell, or no measured count reaches it (no extrapolation).
        reached = df[df.peak_over_noise >= detection_threshold_sd].groupby('cell').n_spikes.min()
        wide['min_detectable_spikes'] = reached.reindex(wide.index)
        g = df.groupby('n_spikes')
        log(f'QC per cell (>= {min_events_per_cell} isolated events): '
            + ', '.join(f'{n} spk: {int(g.size()[n])} cells, median peak {g.peak_dff_pct.median()[n]:.0f}% = '
                        f'{g.peak_over_noise.median()[n]:.1f} noise SD' for n in g.size().index))
        mds = wide.min_detectable_spikes
        log(f'QC min_detectable_spikes (peak >= {detection_threshold_sd} noise SD): '
            + ', '.join(f'{int(n)}: {int((mds == n).sum())}' for n in sorted(mds.dropna().unique()))
            + f', not reached: {int(mds.isna().sum())} cells')
        wide.to_csv(os.path.join(save_folder, 'per_cell_spike_amplitude.csv'))
        np.save(os.path.join(save_folder, 'min_detectable_spikes.npy'),
                reached.reindex(np.arange(traces.shape[0])).to_numpy(dtype=float))

        if noise_rel is not None:
            ns_all = [n for n in n_spikes_to_plot]
            counts = [int((wide.min_detectable_spikes == n).sum()) for n in ns_all]
            not_reached = int(wide.min_detectable_spikes.isna().sum())
            fig, ax = plt.subplots(figsize=(1.2 * len(ns_all) + 2.5, 4))
            labels = [str(n) for n in ns_all] + ['not\nreached']
            bars = ax.bar(labels, counts + [not_reached], color=['tab:blue'] * len(ns_all) + ['grey'])
            total = sum(counts) + not_reached
            for b, c in zip(bars, counts + [not_reached]):
                ax.annotate(f'{c}\n({100 * c / total:.0f}%)', (b.get_x() + b.get_width() / 2, c), ha='center',
                            va='bottom', fontsize=8)
            ax.set_xlabel(f'min_detectable_spikes (isolated-event peak >= {detection_threshold_sd} x noise_rel)')
            ax.set_ylabel('cells')
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f'{session_id} - {total} cells with isolated events (>= {min_events_per_cell} per n)',
                         fontsize=9)
            save_figure(fig, save_folder, 'min_detectable_spikes')

        ns = [n for n in n_spikes_to_plot if (df.n_spikes == n).any()]
        fig, axes = plt.subplots(1, 2, figsize=(4 + 1.1 * len(ns), 4.5), squeeze=False)
        for ax, col, label in [(axes[0, 0], 'peak_dff_pct', 'Peak ΔdF/F (%) of the cell average'),
                               (axes[0, 1], 'peak_over_noise', 'Peak / noise_rel (noise SDs)')]:
            data = [df.loc[df.n_spikes == n, col].dropna().to_numpy() for n in ns]
            ax.boxplot(data, labels=[f'{n}\n({len(x)})' for n, x in zip(ns, data)], showfliers=True,
                       flierprops=dict(markersize=3))
            ax.set_xlabel('Spikes in isolated frame (cells)')
            ax.set_ylabel(label)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0, 1].axhline(3, color='grey', ls='--', lw=0.8)
        fig.suptitle(f'{session_id} - per-cell peak response (cells with >= {min_events_per_cell} events)',
                     fontsize=9)
        save_figure(fig, save_folder, 'dff_peak_by_n_spikes_per_cell')

    # Example df/f traces with rate and spike count, cells passing the SNR cut (snr.npy from compute_dff_updated)
    if neuron_snr is not None and example_snr_min is not None:
        candidates = np.flatnonzero(neuron_snr >= example_snr_min)
        snr_label = f'SNR >= {example_snr_min}'
    else:
        candidates = np.arange(traces.shape[0])
        snr_label = 'no SNR cut'
    if excluded_cells is not None:
        candidates = candidates[~excluded_cells[candidates]]
    if example_first_cells:
        example_cells = candidates[:n_example_cells]
    else:
        example_cells = rng.choice(candidates, size=min(n_example_cells, candidates.size), replace=False)
    log(f'QC example cells ({snr_label}): {list(example_cells)}')
    if example_cells.size == 0:
        return

    # plot with scale in % for df/f, rate is in spks/s, spike count as vertical lines. Param for t start and length
    f0 = int(round(example_t_start * imaging_rate))
    f1 = min(traces.shape[1], f0 + int(round(example_t_length * imaging_rate)))
    t_ex = frame_timestamps[f0:f1]
    fig, axes = plt.subplots(4 * len(example_cells), 1, figsize=(10, 3 * len(example_cells)), sharex=True,
                             gridspec_kw={'height_ratios': [2, 3, 2, 1] * len(example_cells)}, squeeze=False)
    axes = axes[:, 0]
    count_vmax = max(1, int(spikes[example_cells, f0:f1].max()))
    for i, cell in enumerate(example_cells):
        ax_raw, ax_dff, ax_rate, ax_count = axes[4 * i:4 * i + 4]
        ax_raw.plot(t_ex, raw_traces[cell, f0:f1], color='tab:green', lw=0.8)
        ax_raw.set_ylabel('F raw\n(a.u.)')
        snr_txt = f' - SNR {neuron_snr[cell]:.1f}' if neuron_snr is not None else ''
        ax_raw.set_title(f'Cell {cell}{snr_txt}', fontsize=9, loc='left')
        ax_dff.plot(t_ex, traces[cell, f0:f1], color='k', lw=0.8)
        ax_dff.set_ylabel('dF/F (%)')
        ax_rate.plot(t_ex, rate[cell, f0:f1], color='tab:blue', lw=0.8)
        ax_rate.set_ylabel('Rate\n(spk/s)')
        # Spike count per frame as vertical lines (height = count)
        counts = spikes[cell, f0:f1]
        ax_count.vlines(t_ex[counts > 0], 0, counts[counts > 0], color='tab:red', lw=0.8)
        ax_count.set_ylim(0, count_vmax)
        ax_count.set_yticks([0, count_vmax])
        ax_count.set_ylabel('Spk /\nframe')
        for ax in (ax_raw, ax_dff, ax_rate, ax_count):
            ax.spines[['top', 'right']].set_visible(False)
    # Artifact periods shaded on every panel.
    if artifacts is not None and artifacts[f0:f1].any():
        edges = np.diff(np.r_[0, artifacts[f0:f1].astype(int), 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            for ax in axes:
                ax.axvspan(t_ex[a], t_ex[b - 1], color='grey', alpha=0.25, lw=0)
    axes[-1].set_xlabel('Time from imaging start (s)' + ('  (grey: artifacts)' if artifacts is not None else ''))
    fig.suptitle(f'{session_id} - example cells ({snr_label})')
    save_figure(fig, save_folder, 'example_cells_f_dff_rate_spikes')


def process_suite2p_folder(suite2p_folder, model_name, model_folder, frame_rate, cascade_funcs, overwrite=False,
                           spike_exclusion_pad=0.0, neuropil_ratio_threshold=0.8, contamination_threshold=0.8,
                           **quant_params):
    # Cells with neuropil_ratio_max > neuropil_ratio_threshold (F_cor baseline collapsed, at least for one
    # 60 s window: exploding dF/F) or
    # contamination > contamination_threshold are left out of the quantification figures. Their spikes are
    # still inferred and saved.
    # spike_exclusion_pad: spikes within this many seconds of an artifact frame are left out of the
    # quantification (CASCADE looks at +-32 frames, so nearby predictions can be affected).
    tqdm.write(f'\nProcessing {suite2p_folder}.')
    t_start = time.time()

    def log(msg):
        # Step message with time elapsed since the start of the session.
        tqdm.write(f'  [{time.time() - t_start:6.1f} s] {msg}')

    save_folder = os.path.join(suite2p_folder, model_name)
    os.makedirs(save_folder, exist_ok=True)
    # <mouse>/<session>/suite2p/plane0 -> session id
    session_id = os.path.basename(os.path.dirname(os.path.dirname(suite2p_folder)))

    log('Loading dff.npy and F_raw.npy.')
    dff = np.load(os.path.join(suite2p_folder, 'dff.npy'))
    if np.isnan(dff).any():
        raise ValueError(f'{suite2p_folder}/dff.npy contains NaN; CASCADE needs the unmasked dff.')
    F_raw = np.load(os.path.join(suite2p_folder, 'F_raw.npy'))
    snr_path = os.path.join(suite2p_folder, 'snr.npy')
    neuron_snr = np.load(snr_path) if os.path.exists(snr_path) else None
    if neuron_snr is not None and neuron_snr.size != dff.shape[0]:
        log(f'snr.npy has {neuron_snr.size} values for {dff.shape[0]} cells, ignored.')
        neuron_snr = None
    noise_path = os.path.join(suite2p_folder, 'noise_rel.npy')
    noise_rel = np.load(noise_path) if os.path.exists(noise_path) else None
    if noise_rel is not None and noise_rel.size != dff.shape[0]:
        noise_rel = None
    log(f'{dff.shape[0]} cells x {dff.shape[1]} frames ({dff.shape[1] / frame_rate / 60:.1f} min at {frame_rate} Hz).')

    # Artifact frames from compute_dff_updated (suite2p badframes + motion), absent for older outputs.
    artifacts, dff_masked, unreliable = None, None, None
    art_path = os.path.join(suite2p_folder, 'artifact_frames.npy')
    if os.path.exists(art_path):
        artifacts = np.load(art_path).astype(bool)
        if artifacts.size != dff.shape[1]:
            log(f'artifact_frames.npy has {artifacts.size} frames for {dff.shape[1]}, ignored.')
            artifacts = None
    if artifacts is not None and artifacts.any():
        dff_masked = np.load(os.path.join(suite2p_folder, 'dff_masked.npy'))
        pad = int(round(spike_exclusion_pad * frame_rate))
        unreliable = window_any(artifacts, pad, pad)
        log(f'Artifact frames: {100 * artifacts.mean():.1f}% | with +-{spike_exclusion_pad} s padding: '
            f'{100 * unreliable.mean():.1f}% of frames left out of the quantification.')

    spike_prob, spikes = run_inference(dff, model_name, model_folder, frame_rate, save_folder, *cascade_funcs,
                                       overwrite=overwrite, log=log, dff_noise=dff_masked)
    del dff_masked

    excluded_cells = None
    # neuropil_ratio_max (worst 60 s window) when available, else the session neuropil_ratio (older outputs).
    ratio_metric = ('neuropil_ratio_max' if os.path.exists(os.path.join(suite2p_folder, 'neuropil_ratio_max.npy'))
                    else 'neuropil_ratio')
    for metric, threshold in [(ratio_metric, neuropil_ratio_threshold),
                              ('contamination', contamination_threshold)]:
        path = os.path.join(suite2p_folder, f'{metric}.npy')
        if threshold is None or not os.path.exists(path):
            continue
        values = np.load(path)
        if values.size != dff.shape[0]:
            continue
        above = values > threshold
        excluded_cells = above if excluded_cells is None else excluded_cells | above
        log(f'{above.sum()} cells with {metric} > {threshold}: {np.flatnonzero(above)}')
    if excluded_cells is not None:
        log(f'{excluded_cells.sum()} cells left out of the quantification.')

    log('Plotting quantification figures.')
    # CASCADE output is expected spikes per frame: x frame rate gives spk/s.
    quantify(F_raw, dff * 100, spike_prob * frame_rate, spikes, frame_rate, save_folder, session_id,
             neuron_snr=neuron_snr, artifacts=artifacts, unreliable=unreliable, excluded_cells=excluded_cells,
             noise_rel=noise_rel, log=log, **quant_params)
    log(f'Done. Data saved: {save_folder}')


# Keys of the config's cascade: block passed to quantify.
QUANT_PARAMS = ['gap_s', 'n_spikes_to_plot', 't_before', 't_after', 'clear_before', 'clear_after',
                'n_single_examples', 'n_example_cells', 'example_snr_min', 'example_first_cells',
                'example_t_start', 'example_t_length', 'seed', 'min_events_per_cell', 'detection_threshold_sd']


def run_from_config(config):
    # Spike inference + quantification for all requested mice/sessions, from the cascade: block.
    cfg = config.get('cascade') or {}
    dff_cfg = config.get('dff_updated') or {}  # QC thresholds shared with compute_dff_updated
    layout = config.get('folder_layout') or {}
    unknown = set(cfg) - set(QUANT_PARAMS) - {'cascade_repo', 'model_name', 'frame_rate', 'overwrite',
                                               'spike_exclusion_pad'}
    if unknown:
        raise KeyError(f'Unknown cascade options {sorted(unknown)}.')
    cascade_repo = os.path.abspath(cfg['cascade_repo'])
    model_name = cfg['model_name']
    frame_rate = cfg.get('frame_rate') or (config.get('suite2p_ops') or {}).get('fs')
    if not frame_rate:
        raise KeyError('Set cascade.frame_rate (or suite2p_ops.fs) in the config.')

    # Resolve folders before import_cascade changes the working directory.
    suite2p_folders = find_suite2p_folders(
        experimenter=config['experimenter'],
        mice_ids=config['mice'],
        haas_path=config.get('on_haas', False),
        suite2p_subpath=tuple(layout.get('suite2p_subpath', ('suite2p', 'plane0'))),
        paths=config.get('paths'),
        experimenter_map=config.get('experimenter_map'),
    )
    print(suite2p_folders)

    cascade_funcs = import_cascade(cascade_repo)
    model_folder = os.path.join(cascade_repo, 'Pretrained_models')
    if not os.path.isfile(os.path.join(model_folder, model_name, 'config.yaml')):
        cascade_funcs[0].download_model(model_name, model_folder=model_folder, verbose=1)
    params = {k: cfg[k] for k in QUANT_PARAMS if k in cfg}
    for suite2p_folder in tqdm(suite2p_folders, desc='CASCADE'):
        process_suite2p_folder(suite2p_folder, model_name, model_folder, frame_rate, cascade_funcs,
                               overwrite=cfg.get('overwrite', False),
                               spike_exclusion_pad=cfg.get('spike_exclusion_pad', 0.0),
                               neuropil_ratio_threshold=dff_cfg.get('neuropil_ratio_threshold', 0.8),
                               contamination_threshold=dff_cfg.get('contamination_threshold', 0.8), **params)


if __name__ == '__main__':
    # Print each line immediately, also when piped (tee) or run in tmux.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        description='CASCADE spike inference on dff.npy and quantification figures. '
                    'Run in the Cascade environment (see README).')
    parser.add_argument('--config', '-c', required=True, help='Path to a YAML run config (uses its cascade: block).')
    args = parser.parse_args()
    run_from_config(load_config(os.path.abspath(args.config)))
