import argparse
import os
import sys
import time
import scipy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted

from config import get_experimenter_analysis_folder, load_config


def set_merged_roi_to_non_cell(stat, iscell):
    # Set merged cells to 0 in iscell.
    if 'inmerge' in stat[0].keys():
        print('Cells in merge')
        for i, st in enumerate(stat):
            # 0: no merge; -1: input of a merge; index > 0: result of a merge.
            if st['inmerge'] not in [0, -1]:
                iscell[i][0] = 0.0

    return iscell


def get_artifact_mask(badframes, fs, merge_gap=2, pad=2):
    '''
    Turn flagged frames into artifact periods: flagged frames closer than
    merge_gap (s) are merged into one artifact, and each artifact is extended
    by pad (s) on both sides.
    Returns the boolean frame mask and the number of artifacts.
    '''
    mask = np.asarray(badframes).astype(bool)
    if mask.any():
        gap = max(1, int(round(merge_gap * fs)))
        mask = scipy.ndimage.binary_closing(np.pad(mask, gap), structure=np.ones(gap + 1))[gap:-gap]
        mask = scipy.ndimage.binary_dilation(mask, structure=np.ones(2 * int(round(pad * fs)) + 1))
    _, n_artifacts = scipy.ndimage.label(mask)

    return mask, n_artifacts


def load_corrxy(suite2p_folder, ops):
    # Per-frame phase correlation of each frame with the registration reference
    # image. reg_outputs.npy for recent suite2p, ops.npy for older versions.
    reg_path = os.path.join(suite2p_folder, 'reg_outputs.npy')
    if os.path.exists(reg_path):
        reg = np.load(reg_path, allow_pickle=True).item()
        if 'corrXY' in reg:
            return np.asarray(reg['corrXY'], dtype=float)
    if 'corrXY' in ops:
        return np.asarray(ops['corrXY'], dtype=float)
    return None


def get_motion_frames(corrXY, fs, threshold=0.7, window=20):
    '''
    Frames whose correlation with the registration reference (corrXY) drops
    below threshold x its rolling median over window (s). Catches uncorrected
    motion (z-motion, large xy jumps) that suite2p badframes miss: its
    th_badframes is normalized by the mean shift of the session, so frequent
    jumps never stand out.
    '''
    local = scipy.ndimage.median_filter(corrXY, size=int(round(window * fs)) | 1, mode='nearest')
    return corrXY < threshold * local


def get_brightness_step_frames(F_raw, fs, threshold=1.5, window=60):
    '''
    Frames affected by sudden whole-FOV brightness steps (e.g. immersion water
    refilled): population median F changing by more than threshold x between
    the 1 s before and the 1 s after a frame. The rolling F0 (centered, window
    s, low percentile) follows the dim side of a step and inflates every
    cell's dF/F on the bright side for about window / 2: after an upward step
    (water refilled), before a downward one (e.g. PMT settling at the start).
    That side is masked, plus 1 s on the other side.
    Returns the boolean frame mask and the step times (s).
    '''
    pop = np.median(F_raw, axis=0).astype(float)
    w = max(1, int(round(fs)))
    csum = np.r_[0, np.cumsum(pop)]
    i = np.arange(w, pop.size - w)
    ratio = ((csum[i + w] - csum[i]) / w) / ((csum[i] - csum[i - w]) / w)
    big = i[(ratio > threshold) | (ratio < 1 / threshold)]
    mask = np.zeros(pop.size, dtype=bool)
    steps = []
    for group in np.split(big, np.flatnonzero(np.diff(big) > 5 * w) + 1) if big.size else []:
        step = group[np.argmax(np.abs(np.log(ratio[group - w])))]
        half = int(round(window / 2 * fs))
        if ratio[step - w] > 1:  # brighter after: F0 lags after the step
            mask[max(0, step - w):step + half] = True
        else:                    # dimmer after: F0 is pulled down before the step
            mask[max(0, step - half):step + w] = True
        steps.append(step / fs)
    return mask, steps


def compute_motion_r(traces, corrXY, fs, detrend=10):
    '''
    Per-cell correlation between traces (dff: the analysed trace, after
    neuropil subtraction, which can amplify or cancel an artifact) and corrXY,
    slow drift removed (moving average over detrend s). ~0: trace unaffected
    by motion. Large |r| (either sign): it follows registration quality frame
    by frame, i.e. a structure moving in or out of the ROI. Strong population
    activity also lowers corrXY, so moderate values can come from real activity.
    '''
    size = int(round(detrend * fs)) | 1
    F = traces.astype(np.float32)
    F -= scipy.ndimage.uniform_filter1d(F, size, axis=1, mode='nearest')
    F -= F.mean(axis=1, keepdims=True)
    c = corrXY - scipy.ndimage.uniform_filter1d(corrXY, size, mode='nearest')
    c -= c.mean()
    return (F @ c) / np.sqrt((F ** 2).sum(axis=1) * (c ** 2).sum())


def compute_baseline(F, fs, window=60, percentile=8, sigma=1, bad_frames=None):
    '''
    F0 as a low percentile of F in a rolling window.
    F: (n_cells, n_frames)
    fs: sampling frequency (Hz)
    window: total rolling window length (s)
    percentile: percentile taken as baseline
    sigma: std (s) of the gaussian smoothing applied before the percentile.
           Without it the low percentile tracks the noise floor and F0 is
           biased low, which inflates dff.
    bad_frames: boolean mask of frames ignored for F0 (motion artifacts).
    '''
    if bad_frames is None or not bad_frames.any():
        F_smooth = scipy.ndimage.gaussian_filter1d(F, sigma=sigma * fs, axis=1, mode='reflect')
    else:
        # Smooth with bad frames excluded (normalized convolution), so the
        # artifact does not leak into neighbouring frames, then mask them.
        good = (~bad_frames).astype(float)
        num = scipy.ndimage.gaussian_filter1d(F * good, sigma=sigma * fs, axis=1, mode='reflect')
        den = scipy.ndimage.gaussian_filter1d(good, sigma=sigma * fs, mode='reflect')
        with np.errstate(invalid='ignore', divide='ignore'):
            F_smooth = num / den
        F_smooth[:, bad_frames] = np.nan

    # Full-rate rolling percentile. pandas is used because it is much faster
    # than scipy.ndimage.percentile_filter for long windows; it skips NaNs.
    # The window shrinks at the edges of the recording.
    size = int(round(window * fs)) | 1  # Odd size so the window is centered.
    F0 = pd.DataFrame(F_smooth.T).rolling(size, center=True, min_periods=1).quantile(percentile / 100)

    return F0.to_numpy().T


def compute_dff(F_raw, F_neu, fs, neuropil_coef=0.7, window=60, percentile=8, sigma=1, min_f0=1, bad_frames=None):
    '''
    F_raw: raw traces (suite2p F)
    F_neu: neuropil traces (suite2p Fneu)
    fs: sampling frequency
    bad_frames: boolean mask of artifact frames, excluded from F0.
    Returns F_cor, F0 and dff = (F_cor - F0) / F0, with F0 computed on F_cor.
    F_cor and dff keep their values on bad frames, mask them with mask_frames.
    '''
    F_cor = F_raw - neuropil_coef * F_neu
    F0 = compute_baseline(F_cor, fs, window=window, percentile=percentile, sigma=sigma, bad_frames=bad_frames)
    F0 = np.maximum(F0, min_f0)  # Avoid division by ~0 / negative baselines.
    dff = (F_cor - F0) / F0

    return F_cor, F0, dff


def mask_frames(x, bad_frames):
    # Copy of x with bad frames set to NaN.
    x = np.array(x, dtype=np.result_type(x, np.float32), copy=True)
    x[:, bad_frames] = np.nan
    return x


def frame_noise(x):
    '''
    Per-cell estimate of the noise std of a single frame, in the units of x
    (a.u. for F_cor, dF/F0 for dff).

    Steps:
    1. diff: x[t+1] - x[t]. Calcium signal (transients, drift) changes little
       between consecutive frames (33 ms at 30 Hz), so it mostly cancels out
       and what remains is frame-to-frame noise. If the noise has std sigma
       and is independent across frames, the difference has std sigma * sqrt(2).
    2. median of |diff|: robust to the few large jumps at transient onsets,
       which would inflate a std.
    3. / 0.6745: for gaussian noise, median(|value|) = 0.6745 * std, so this
       converts the median back into a std.
    4. / sqrt(2): undoes the sqrt(2) from step 1, giving the single-frame std.

    NaN frames (artifacts) are ignored. Assumes noise independent between
    frames: temporally smoothed traces give an underestimate.
    '''
    return np.nanmedian(np.abs(np.diff(x, axis=1)), axis=1) / (0.6745 * np.sqrt(2))


def compute_noise_metrics(F_cor, dff, snr_percentile=99):
    '''
    Per-cell noise metrics.
    noise_abs: frame noise of F_cor (a.u.)
    noise_rel: frame noise of dff (dF/F0 units, i.e. noise relative to baseline)
    snr: high percentile of dff (large transients) / noise_rel. 99th rather
         than 99.9th: set by the top 1% of frames (~1,500 in a session) instead
         of the top 0.1% (~150), so a few extreme events weigh less. A pure-noise
         trace gives ~2.3.
    '''
    noise_abs = frame_noise(F_cor)
    noise_rel = frame_noise(dff)
    snr = np.nanpercentile(dff, snr_percentile, axis=1) / noise_rel  # NaN frames (artifacts, collapse) ignored

    return pd.DataFrame({'noise_abs': noise_abs, 'noise_rel': noise_rel, 'snr': snr})


def compute_contamination_metrics(F_raw, F_neu, fs, neuropil_coef=0.7, bad_frames=None, band=(0.5, 30),
                                  ratio_window=60, collapse_threshold=None):
    '''
    Per-cell neuropil contamination metrics.
    neuropil_ratio: neuropil_coef * median(Fneu) / median(F), share of the
                    baseline removed by neuropil subtraction (>= 1: F_cor <= 0).
    neuropil_ratio_max: the same ratio in non-overlapping ratio_window (s)
                    windows, worst window. Catches cells whose baseline
                    collapses for a while (e.g. their neuropil ring brightens
                    for minutes), which the session median misses: F0 then
                    drops to its floor and dF/F explodes. Windows with less
                    than half good frames are skipped.
    neuropil_f_rho: correlation of F and Fneu fluctuations, band-passed between
                   band[0] and band[1] seconds (removes frame noise and slow drift).
    contamination: neuropil_ratio * neuropil_f_rho. ~0: clean, > ~0.8: F is
                   mostly neuropil (likely not a usable cell).
    Bad frames are excluded.
    If collapse_threshold is given, also returns a (n_cells, n_frames) bool
    mask of the windows where a cell's ratio exceeds it (baseline collapsed).
    '''
    good = slice(None) if bad_frames is None else ~bad_frames
    ratio = neuropil_coef * np.median(F_neu[:, good], axis=1) / np.median(F_raw[:, good], axis=1)

    win = int(round(ratio_window * fs))
    n_win = F_raw.shape[1] // win
    ratio_max = np.full(F_raw.shape[0], np.nan)
    collapse = np.zeros(F_raw.shape, dtype=bool) if collapse_threshold is not None else None
    if n_win:
        ok = np.ones(n_win * win, dtype=bool) if bad_frames is None else ~bad_frames[:n_win * win]
        windows = [np.flatnonzero(ok[w * win:(w + 1) * win]) + w * win for w in range(n_win)]
        windows = [w for w in windows if w.size >= win / 2]
        if windows:
            local = np.stack([neuropil_coef * np.median(F_neu[:, w], axis=1) / np.median(F_raw[:, w], axis=1)
                              for w in windows], axis=1)
            ratio_max = local.max(axis=1)
            if collapse is not None:
                for k, w in enumerate(windows):
                    cells = local[:, k] > collapse_threshold
                    if cells.any():
                        collapse[np.ix_(cells, np.arange(w[0], w[-1] + 1))] = True

    def bandpass(x):
        x = x.astype(np.float32)
        return (scipy.ndimage.gaussian_filter1d(x, band[0] * fs, axis=1)
                - scipy.ndimage.gaussian_filter1d(x, band[1] * fs, axis=1))[:, good]

    Fb, Nb = bandpass(F_raw), bandpass(F_neu)
    Fb -= Fb.mean(axis=1, keepdims=True)
    Nb -= Nb.mean(axis=1, keepdims=True)
    corr = (Fb * Nb).sum(axis=1) / np.sqrt((Fb ** 2).sum(axis=1) * (Nb ** 2).sum(axis=1))

    metrics = pd.DataFrame({'neuropil_ratio': ratio, 'neuropil_ratio_max': ratio_max, 'neuropil_f_rho': corr,
                            'contamination': ratio * corr})
    return metrics if collapse is None else (metrics, collapse)


def select_example_cells(metrics, mode='quality', n_cells=6, neuropil_ratio_threshold=0.8,
                         contamination_threshold=0.8):
    '''
    Cells shown in dff_check. Returns (cells, labels).
    mode='quality': n_cells // 3 good, median and bad cells. Quality score =
        mean of the percentile ranks of snr (high = good) and contamination
        (low = good). Cells with neuropil_ratio_max (neuropil_ratio for older
        metrics) above neuropil_ratio_threshold or contamination above
        contamination_threshold (flagged) are never good or median, and come
        first among the bad ones: a collapsed baseline inflates snr and keeps
        contamination low, so the score alone ranks them as good.
        Ordered good -> median -> bad.
    mode='random': random cells. mode='first': the first n_cells.
    '''
    n_total = len(metrics)
    n_cells = min(n_cells, n_total)
    if mode == 'random':
        cells = np.sort(np.random.default_rng().choice(n_total, size=n_cells, replace=False))
        return cells, [''] * n_cells
    if mode == 'first':
        return np.arange(n_cells), [''] * n_cells

    score = (metrics.snr.rank(pct=True) + (1 - metrics.contamination.rank(pct=True))).to_numpy() / 2
    ratio = metrics.neuropil_ratio_max if 'neuropil_ratio_max' in metrics else metrics.neuropil_ratio
    flagged = ((ratio > neuropil_ratio_threshold) | (metrics.contamination > contamination_threshold)).to_numpy()
    order = np.argsort(-score)  # best first
    kept, worst = order[~flagged[order]], order[flagged[order]][::-1]
    k = max(1, n_cells // 3)
    mid = len(kept) // 2
    good, median = kept[:k], kept[mid - k // 2: mid - k // 2 + k]
    # Bad: flagged cells first (worst score first), then the worst of the others.
    bad = np.concatenate([worst, kept[::-1]])[:k][::-1]
    cells = np.concatenate([good, median, bad])
    labels = ['good'] * len(good) + ['median'] * len(median) + ['bad'] * len(bad)
    return cells, labels


def plot_noise_distributions(metrics, save_path, cells=None, contamination_threshold=0.8, neuropil_ratio_threshold=0.8,
                             neuropil_coef=0.7, artifacts=None, fs=None, title=None,
                             snr_label='SNR (99th pct of dF/F0 / noise rel)'):
    '''
    One row per concern (noise_abs is saved but not plotted):
    Noise / signal: histograms of noise_rel and snr.
    Neuropil: contamination histogram, neuropil ratio vs neuropil_f_rho colored
        by SNR, neuropil ratio vs noise_rel, SNR vs contamination.
    Movement: motion_r histogram and, if artifacts (bool per frame) and fs are
        given, the distribution of artifact period lengths with the share of
        the session masked.
    `cells` (indices) are marked so they can be matched with the dff_check figure.
    '''
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.3, 1])

    def histogram(ax, col, label):
        values = metrics[col].to_numpy()
        if col == 'motion_r':
            xmin, xmax = -1, 1
        else:
            # Cells with a tiny F0 have extreme values: clip the x range at
            # median + 5 IQR so the bulk of the distribution stays readable.
            q25, q50, q75 = np.percentile(values, [25, 50, 75])
            xmin, xmax = values.min(), min(values.max(), q50 + 5 * (q75 - q25))
        n_out = np.sum(values > xmax)
        ax.hist(values, bins=50, range=(xmin, xmax), color='#2a78d6', edgecolor='#fcfcfb', linewidth=0.5)
        ax.axvline(np.median(values), color='#0b0b0b', lw=1.5, ls='--')
        ax.set_title(f'median = {np.median(values):.3g}   n = {len(values)} cells'
                     + (f'   ({n_out} > {xmax:.3g} not shown)' if n_out else ''),
                     fontsize=8, color='#52514e')
        if cells is not None:
            # Staggered labels so close cells don't overlap.
            for k, icell in enumerate(sorted(cells, key=lambda c: values[c])):
                x = min(values[icell], xmax)
                ax.axvline(x, color='#eb6834', lw=1)
                ax.text(x, 0.97 - 0.08 * (k % 4), f' {icell}', transform=ax.get_xaxis_transform(),
                        ha='left', va='top', fontsize=7, color='#52514e')
            ax.plot([], [], color='#eb6834', lw=1, label='cells in dff_check')
            ax.plot([], [], color='#0b0b0b', lw=1.5, ls='--', label='median')
            ax.legend(loc='center right', frameon=False, fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel('cells')
        ax.spines[['top', 'right']].set_visible(False)

    # Row 1: noise / signal.
    ax_noise, ax_snr = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:])
    histogram(ax_noise, 'noise_rel', 'relative noise (dF/F0)')
    histogram(ax_snr, 'snr', snr_label)

    # Row 2: neuropil.
    ax_cont, ax_rc, ax_rn, ax_cs = (fig.add_subplot(gs[1, i]) for i in range(4))
    histogram(ax_cont, 'contamination', 'contamination (neuropil ratio x corr)')
    ax_cont.axvline(contamination_threshold, color='#e34948', lw=1, ls=':')
    ratio, corr = metrics.neuropil_ratio.to_numpy(), metrics.neuropil_f_rho.to_numpy()
    contamination, snr = metrics.contamination.to_numpy(), metrics.snr.to_numpy()
    high = contamination > contamination_threshold

    # Row 3: movement.
    plot_motion = 'motion_r' in metrics and metrics.motion_r.notna().any()
    plot_artifacts = artifacts is not None and fs is not None
    if plot_motion:
        histogram(fig.add_subplot(gs[2, :2]), 'motion_r', 'motion_r  corr(dF/F0, corrXY), drift removed')
    if plot_artifacts:
        # Artifact periods: length distribution and share of the session masked.
        ax_art = fig.add_subplot(gs[2, 2:])
        labels, n_periods = scipy.ndimage.label(artifacts)
        lengths = np.bincount(labels.ravel())[1:] / fs
        total = artifacts.sum()
        art_title = (f'{n_periods} artifact periods, {total} frames\n'
                 f'= {100 * total / artifacts.size:.1f}% of session ({total / fs / 60:.1f} min)')
        if n_periods:
            ax_art.hist(lengths, bins=np.logspace(np.log10(lengths.min()), np.log10(lengths.max() + 1e-9), 30),
                        color='#898781', edgecolor='#fcfcfb', linewidth=0.5)
            ax_art.set_xscale('log')
            ax_art.axvline(np.median(lengths), color='#0b0b0b', lw=1.5, ls='--')
            art_title += f'\nmedian length {np.median(lengths):.2f} s'
        ax_art.set_title(art_title, fontsize=8, color='#52514e')
        ax_art.set_xlabel('artifact period length (s)')
        ax_art.set_ylabel('periods')
        ax_art.spines[['top', 'right']].set_visible(False)

    sc = ax_rc.scatter(ratio, corr, c=np.log10(np.clip(snr, 1, None)), s=14, cmap='viridis')
    ax_rc.scatter(ratio[high], corr[high], s=50, facecolors='none', edgecolors='#e34948',
                  label=f'contamination > {contamination_threshold} ({high.sum()} cells)')
    ax_rc.axvline(neuropil_ratio_threshold, color='#e34948', lw=1, ls='--',
                  label=f'neuropil ratio > {neuropil_ratio_threshold} ({(ratio > neuropil_ratio_threshold).sum()} cells)')
    if 'neuropil_ratio_max' in metrics:
        # Cells whose worst 60 s window crosses the threshold although the session ratio does not.
        local = (metrics.neuropil_ratio_max.to_numpy() > neuropil_ratio_threshold) & (ratio <= neuropil_ratio_threshold)
        ax_rc.scatter(ratio[local], corr[local], s=60, marker='s', facecolors='none', edgecolors='#eb6834', lw=1.2,
                      label=f'only neuropil_ratio_max > {neuropil_ratio_threshold} ({local.sum()} cells)')
    fig.colorbar(sc, ax=ax_rc, label='log10 SNR')
    ax_rc.set_xlabel(f'neuropil ratio  {neuropil_coef:g}·Fneu / F (session baseline)')
    ax_rc.set_ylabel('neuropil_f_rho  corr(F, Fneu), 0.5-30 s')

    # Relative noise vs neuropil ratio: neuropil subtraction lowers F0 but keeps the neuropil's photon noise,
    # so noise relative to the baseline is expected to grow with the share of the baseline removed.
    noise_rel = metrics.noise_rel.to_numpy()
    ax_rn.scatter(ratio, noise_rel, s=14, color='#2a78d6')
    ax_rn.set_yscale('log')
    ax_rn.axvline(neuropil_ratio_threshold, color='#e34948', lw=1, ls='--')
    rho = pd.Series(ratio).corr(pd.Series(noise_rel), method='spearman')
    ax_rn.set_title(f'Spearman rho = {rho:.2f}', fontsize=8, color='#52514e')
    ax_rn.set_xlabel(f'neuropil ratio  {neuropil_coef:g}·Fneu / F (session baseline)')
    ax_rn.set_ylabel('relative noise (dF/F0)')

    ax_cs.scatter(contamination, snr, s=14, color='#2a78d6')
    ax_cs.set_yscale('log')
    ax_cs.axvline(contamination_threshold, color='#e34948', lw=1, ls=':')
    ax_cs.set_xlabel('contamination (neuropil ratio x corr)')
    ax_cs.set_ylabel('SNR')

    if cells is not None:
        for ax, (x, y) in [(ax_rc, (ratio, corr)), (ax_rn, (ratio, noise_rel)), (ax_cs, (contamination, snr))]:
            ax.scatter(x[cells], y[cells], s=60, marker='D', facecolors='none', edgecolors='#eb6834', lw=1.5,
                       label='cells in dff_check')
            for icell in cells:
                ax.annotate(str(icell), (x[icell], y[icell]), xytext=(5, 5), textcoords='offset points',
                            fontsize=8, color='#0b0b0b')
    for ax in (ax_rc, ax_rn, ax_cs):
        ax.legend(frameon=False, fontsize=7, loc='upper left')
        ax.spines[['top', 'right']].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0.025, 0, 1, 1))
    # Row labels in the left margin, centred on each row.
    rows = [(0, 'NOISE / SIGNAL'), (1, 'NEUROPIL')] + ([(2, 'MOVEMENT')] if plot_motion or plot_artifacts else [])
    for row, name in rows:
        box = gs[row, 0].get_position(fig)
        fig.text(0.006, (box.y0 + box.y1) / 2, name, rotation=90, ha='left', va='center', fontsize=12,
                 fontweight='bold', color='#52514e')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_dff_check(F_raw, F_neu, F_cor, F0, dff, fs, save_path, cells, metrics=None, bad_frames=None,
                   duration=300, seed=None, cell_labels=None, corrXY=None, motion_threshold=None, motion_window=20,
                   title=None):
    '''
    Plot F, Fneu, F_cor, F0 (left) and dff (right) for the given cells
    over a random segment of `duration` seconds. If metrics is given, the
    SNR, contamination and motion_r of each cell are written above its dff
    trace. Artifact periods (bad_frames) are shaded. If corrXY is given, it is
    plotted below both columns, with the detection threshold (motion_threshold x
    rolling median over motion_window s) dashed.
    '''
    rng = np.random.default_rng(seed)
    n = int(round(duration * fs))
    start = rng.integers(0, max(1, F_raw.shape[1] - n))
    sl = slice(start, start + n)
    t = np.arange(start, start + F_raw[:, sl].shape[1]) / fs

    traces = [(F_raw, 'F', '#2a78d6', 0.6), (F_neu, 'Fneu', '#eb6834', 0.6),
              (F_cor, 'F_cor', '#1baf7a', 0.6), (F0, 'F0', '#0b0b0b', 1.5)]

    artifacts = []
    if bad_frames is not None:
        labels, _ = scipy.ndimage.label(bad_frames[sl])
        artifacts = [(t[s][0], t[s][-1]) for s in scipy.ndimage.find_objects(labels)]

    n_rows = len(cells) + (corrXY is not None)
    fig, axes = plt.subplots(n_rows, 2, figsize=(26, 2.8 * n_rows), sharex=True, squeeze=False,
                             gridspec_kw={'height_ratios': [1] * len(cells) + [0.7] * (corrXY is not None)})
    if corrXY is not None:
        # Registration correlation with the reference: drops = motion left after registration.
        threshold = None
        if motion_threshold is not None:
            local = scipy.ndimage.median_filter(corrXY, size=int(round(motion_window * fs)) | 1, mode='nearest')
            threshold = motion_threshold * local[sl]
        for ax in axes[-1]:
            for t0, t1 in artifacts:
                ax.axvspan(t0, t1, color='#898781', alpha=0.25, lw=0)
            ax.plot(t, corrXY[sl], color='#0b0b0b', lw=0.6, label='corrXY')
            if threshold is not None:
                ax.plot(t, threshold, color='#e34948', lw=1, ls='--', label=f'{motion_threshold} x local median')
            ax.set_ylabel('corrXY\n(registration)')
            ax.spines[['top', 'right']].set_visible(False)
        axes[-1, 1].legend(loc='lower right', ncol=2, frameon=False, fontsize=8)
    for row, icell in enumerate(cells):
        ax_f, ax_d = axes[row]
        for t0, t1 in artifacts:
            for ax in (ax_f, ax_d):
                ax.axvspan(t0, t1, color='#898781', alpha=0.25, lw=0)
        for x, label, color, lw in traces:
            ax_f.plot(t, x[icell, sl], color=color, lw=lw, label=label)
        ax_d.plot(t, dff[icell, sl], color='#2a78d6', lw=0.6)
        ax_d.axhline(0, color='#898781', lw=0.8, ls='--')
        label = f'{cell_labels[row]}\n' if cell_labels is not None and cell_labels[row] else ''
        ax_f.set_ylabel(f'{label}cell {icell}\nfluo. (a.u.)')
        ax_d.set_ylabel('dF/F0')
        if metrics is not None:
            m = metrics.iloc[icell]
            text = f'SNR = {m.snr:.3g}   contamination = {m.contamination:.2f}'
            if 'motion_r' in m and np.isfinite(m.motion_r):
                text += f'   motion_r = {m.motion_r:.2f}'
            ax_d.text(1.0, 1.0, text, transform=ax_d.transAxes, ha='right', va='bottom', fontsize=9, color='#0b0b0b')
        for ax in (ax_f, ax_d):
            ax.spines[['top', 'right']].set_visible(False)
    if artifacts:
        axes[0, 0].axvspan(np.nan, np.nan, color='#898781', alpha=0.25, lw=0, label='artifact')
    axes[0, 0].legend(loc='upper right', ncol=5, frameon=False, fontsize=8)
    axes[0, 0].set_title('F, Fneu, F_cor, F0', pad=14)
    axes[0, 1].set_title('dF/F0', pad=14)
    axes[-1, 0].set_xlabel('time (s)')
    axes[-1, 1].set_xlabel('time (s)')
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def find_suite2p_folders(experimenter, mice_ids, haas_path=False,
                         suite2p_subpath=('suite2p', 'plane0'),
                         overwrite=False, paths=None, experimenter_map=None):
    # Suite2p output folders of the given mice. Skips folders that already
    # have a dff.npy unless overwrite is set.
    analysis_root = get_experimenter_analysis_folder(experimenter, haas_path, paths, experimenter_map)
    suite2p_folders = []
    for mouse_id in mice_ids:
        mouse_folder = os.path.join(analysis_root, mouse_id)
        for session_id in natsorted(os.listdir(mouse_folder)):
            session_folder = os.path.join(mouse_folder, session_id)
            if not os.path.isdir(session_folder):
                continue
            suite2p_folder = os.path.join(session_folder, *suite2p_subpath)
            if not os.path.isdir(suite2p_folder):
                continue
            if os.path.exists(os.path.join(suite2p_folder, 'dff.npy')) and not overwrite:
                continue
            suite2p_folders.append(suite2p_folder)
    return suite2p_folders


def process_suite2p_folder(suite2p_folder, neuropil_coef=0.7, window=60, percentile=8, sigma=1, min_f0=1,
                           motion_threshold=0.7, motion_window=20,
                           motion_merge_gap=0.2, motion_pad=0.1, brightness_step_threshold=1.5,
                           example_cells='quality', plot_duration=180,
                           contamination_threshold=0.8, neuropil_ratio_threshold=0.8):
    '''
    Compute and save F_cor, F0, dff, artifact frames, noise metrics and QC
    figures for a single suite2p output folder.
    Artifact frames = frames where corrXY < motion_threshold x its rolling
    median over motion_window s (merged motion_merge_gap, padded motion_pad),
    plus whole-FOV brightness steps > brightness_step_threshold x (1 s before
    to window / 2 after each step, where F0 lags). None disables either.
    example_cells: dff_check cells, 'quality' (2 good, 2 median, 2 bad on snr
                   & contamination), 'random' or 'first'.
    '''
    if not os.path.exists(os.path.join(suite2p_folder,'stat.npy')):
        return

    tqdm.write(f'\nProcessing {suite2p_folder}.')
    t_start = time.time()

    def log(msg):
        # Step message with time elapsed since the start of the session.
        tqdm.write(f'  [{time.time() - t_start:6.1f} s] {msg}')

    log('Loading stat.npy, ops.npy, iscell.npy.')
    stat = np.load(os.path.join(suite2p_folder,'stat.npy'), allow_pickle = True)
    ops = np.load(os.path.join(suite2p_folder,'ops.npy'), allow_pickle = True).item()
    iscell = np.load(os.path.join(suite2p_folder,'iscell.npy'), allow_pickle = True)
    log('Loading F.npy.')
    F_raw = np.load(os.path.join(suite2p_folder,'F.npy'), allow_pickle = True)
    log('Loading Fneu.npy.')
    F_neu = np.load(os.path.join(suite2p_folder,'Fneu.npy'), allow_pickle = True)
    log(f'Loaded {F_raw.shape[0]} rois x {F_raw.shape[1]} frames '
        f'({F_raw.shape[1] / ops["fs"] / 60:.1f} min at {ops["fs"]} Hz).')

    # Set merged roi's to non-cells.
    iscell_original = iscell.copy()
    iscell = set_merged_roi_to_non_cell(stat, iscell)
    if not np.array_equal(iscell, iscell_original):
        np.save(os.path.join(suite2p_folder, 'iscell_original'), iscell_original)
        np.save(os.path.join(suite2p_folder, 'iscell'), iscell)
        log('Merged rois set to non-cells, iscell.npy updated.')

    F_raw = F_raw[iscell[:,0]==1.]
    F_neu = F_neu[iscell[:,0]==1.]
    log(f'Kept {F_raw.shape[0]} cells (iscell == 1).')

    # Motion artifacts: drops of the registration correlation (corrXY). suite2p
    # badframes are not used: they all fall inside these periods (checked on
    # RD130/RD132/RD134, 147/147 frames) and would need their own padding.
    bad_frames = np.zeros(F_raw.shape[1], dtype=bool)
    corrXY = load_corrxy(suite2p_folder, ops)
    if corrXY is not None and corrXY.size != F_raw.shape[1]:
        log(f'corrXY has {corrXY.size} frames for {F_raw.shape[1]} imaging frames, ignored.')
        corrXY = None
    if corrXY is None:
        log('No corrXY: motion detection and motion_r skipped, no artifact frames.')
    elif motion_threshold is not None:
        log('Detecting motion artifacts from corrXY.')
        bad_frames, n_motion = get_artifact_mask(
            get_motion_frames(corrXY, fs=ops['fs'], threshold=motion_threshold, window=motion_window),
            fs=ops['fs'], merge_gap=motion_merge_gap, pad=motion_pad)
        log(f'{n_motion} motion periods (corrXY < {motion_threshold} x local median): {bad_frames.sum()} frames '
            f'({100 * bad_frames.mean():.1f}%)')

    # Whole-FOV brightness steps (immersion water refill): F0 lags for window / 2.
    if brightness_step_threshold is not None:
        step_frames, steps = get_brightness_step_frames(F_raw, fs=ops['fs'], threshold=brightness_step_threshold,
                                                        window=window)
        if steps:
            bad_frames = bad_frames | step_frames
            log(f'{len(steps)} brightness step(s) > {brightness_step_threshold}x at '
                f'{", ".join(f"{t:.0f} s" for t in steps)}: {step_frames.sum()} frames masked | '
                f'all artifact frames: {100 * bad_frames.mean():.1f}%')

    log('Computing F_cor, F0 (rolling percentile) and dff.')
    # Without suffix: artifact frames kept. _masked: artifact frames set to NaN.
    F_cor, F0, dff = compute_dff(F_raw, F_neu, fs=ops['fs'], neuropil_coef=neuropil_coef, window=window,
                                 percentile=percentile, sigma=sigma, min_f0=min_f0, bad_frames=bad_frames)
    F_cor_masked = mask_frames(F_cor, bad_frames)
    dff_masked = mask_frames(dff, bad_frames)

    # Cells whose neuropil-corrected baseline is tiny get huge dff values.
    low_f0 = np.median(F0, axis=1) < 0.1 * np.median(F_raw, axis=1)
    if low_f0.any():
        log(f'{low_f0.sum()} cells with F0 < 10% of raw F (dff unreliable): {np.where(low_f0)[0]}')

    # Saving data.
    log('Saving F_raw, F_neu, F_cor, F_cor_masked, F0, dff, dff_masked, artifact_frames.')
    np.save(os.path.join(suite2p_folder, 'F_raw'), F_raw)
    np.save(os.path.join(suite2p_folder, 'F_neu'), F_neu)
    np.save(os.path.join(suite2p_folder, 'F_cor'), F_cor)
    np.save(os.path.join(suite2p_folder, 'F_cor_masked'), F_cor_masked)
    np.save(os.path.join(suite2p_folder, 'F0'), F0)
    np.save(os.path.join(suite2p_folder, 'dff'), dff)
    np.save(os.path.join(suite2p_folder, 'dff_masked'), dff_masked)
    np.save(os.path.join(suite2p_folder, 'artifact_frames'), bad_frames)

    # Noise metrics, on artifact-free traces. roi = index in the full suite2p roi list.
    log('Computing noise and contamination metrics.')
    contamination, collapse = compute_contamination_metrics(
        F_raw, F_neu, fs=ops['fs'], neuropil_coef=neuropil_coef, bad_frames=bad_frames, ratio_window=window,
        collapse_threshold=neuropil_ratio_threshold if neuropil_ratio_threshold is not None else np.inf)
    # Noise and SNR without each cell's collapsed windows (F0 at its floor there inflates the SNR percentile).
    log(f'{collapse.any(axis=1).sum()} cells have collapsed windows: those frames ({100 * collapse.mean():.2f}% of '
        f'all cell x frames) are left out of their noise and SNR; the cells are kept.')
    nan = np.float32(np.nan)  # keeps the float32 copies at half the memory
    metrics = pd.concat([compute_noise_metrics(np.where(collapse, nan, F_cor_masked),
                                               np.where(collapse, nan, dff_masked)), contamination], axis=1)
    del collapse
    metrics['motion_r'] = (compute_motion_r(dff, corrXY, fs=ops['fs']) if corrXY is not None
                           else np.full(F_raw.shape[0], np.nan))
    metrics.insert(0, 'roi', np.where(iscell[:, 0] == 1.)[0])
    metrics.to_csv(os.path.join(suite2p_folder, 'noise_metrics.csv'), index_label='cell')
    # One (n_cells,) array per metric, same cell order as dff.
    for col in ['noise_abs', 'noise_rel', 'snr', 'neuropil_ratio', 'neuropil_ratio_max', 'neuropil_f_rho',
                'contamination', 'motion_r']:
        np.save(os.path.join(suite2p_folder, col), metrics[col].to_numpy())
    log(f'Median noise abs = {metrics.noise_abs.median():.3g} a.u. | noise rel = {metrics.noise_rel.median():.3g} '
        f'| SNR = {metrics.snr.median():.3g} | contamination = {metrics.contamination.median():.2f} '
        f'({(metrics.contamination > contamination_threshold).sum()} cells > {contamination_threshold}) '
        f'| neuropil_ratio_max > {neuropil_ratio_threshold}: '
        f'{(metrics.neuropil_ratio_max > neuropil_ratio_threshold).sum()} cells '
        f'| |motion_r| > 0.5: {(metrics.motion_r.abs() > 0.5).sum()} cells')

    # Figures, same example cells in both.
    log('Plotting dff_check.png and noise_distributions.png.')
    # <mouse>/<session>/suite2p/plane0 -> session id, written on both figures.
    session_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.normpath(suite2p_folder))))
    cells, cell_labels = select_example_cells(metrics, mode=example_cells,
                                              neuropil_ratio_threshold=neuropil_ratio_threshold,
                                              contamination_threshold=contamination_threshold)
    plot_dff_check(F_raw, F_neu, F_cor, F0, dff_masked, fs=ops['fs'], cells=cells, cell_labels=cell_labels,
                   duration=plot_duration, metrics=metrics, corrXY=corrXY, motion_threshold=motion_threshold,
                   motion_window=motion_window, bad_frames=bad_frames, title=session_id,
                   save_path=os.path.join(suite2p_folder, 'dff_check.png'))
    plot_noise_distributions(metrics, cells=cells, contamination_threshold=contamination_threshold,
                             neuropil_ratio_threshold=neuropil_ratio_threshold,
                             artifacts=bad_frames, fs=ops['fs'],
                             neuropil_coef=neuropil_coef, title=session_id,
                             save_path=os.path.join(suite2p_folder, 'noise_distributions.png'))
    log(f'Done. Data saved: {suite2p_folder}')


# Keys of the config's dff_updated: block passed to process_suite2p_folder.
DFF_PARAMS = ['neuropil_coef', 'window', 'percentile', 'sigma', 'min_f0',
              'motion_threshold', 'motion_window', 'motion_merge_gap', 'motion_pad', 'brightness_step_threshold',
              'example_cells', 'plot_duration', 'contamination_threshold', 'neuropil_ratio_threshold']


def run_from_config(config):
    # Compute dff for all requested mice/sessions from a YAML config dict.
    # Parameters are read from the dff_updated: block (the dff: block belongs
    # to compute_dff.py, whose window has a different meaning).
    dff_cfg = config.get('dff_updated') or {}
    layout = config.get('folder_layout') or {}
    unknown = set(dff_cfg) - set(DFF_PARAMS) - {'overwrite'}
    if unknown:
        raise KeyError(f'Unknown dff_updated options {sorted(unknown)}. Known: {DFF_PARAMS + ["overwrite"]}.')

    suite2p_folders = find_suite2p_folders(
        experimenter=config['experimenter'],
        mice_ids=config['mice'],
        haas_path=config.get('on_haas', False),
        suite2p_subpath=tuple(layout.get('suite2p_subpath', ('suite2p', 'plane0'))),
        overwrite=dff_cfg.get('overwrite', False),
        paths=config.get('paths'),
        experimenter_map=config.get('experimenter_map'),
    )
    params = {k: dff_cfg[k] for k in DFF_PARAMS if k in dff_cfg}
    print(suite2p_folders)
    for suite2p_folder in tqdm(suite2p_folders, desc='Processing suite2p folders'):
        process_suite2p_folder(suite2p_folder, **params)


def _legacy_main():
    # The original hardcoded run, kept so usage without --config still works.
    suite2p_folders = find_suite2p_folders('RD', ['RDXXX'], haas_path=False, overwrite=False)
    print(suite2p_folders)
    for suite2p_folder in tqdm(suite2p_folders, desc='Processing suite2p folders'):
        process_suite2p_folder(suite2p_folder, example_cells='quality')


if __name__ == '__main__':
    # Print each line immediately, also when piped (tee) or run in tmux.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        description='Compute dF/F (rolling-percentile F0, artifact masking, QC metrics) '
                    'for Suite2p outputs from a YAML config.')
    parser.add_argument('--config', '-c', default=None,
                        help='Path to a YAML run config (uses its dff_updated: block). If omitted, '
                             'runs the legacy hardcoded settings.')
    args = parser.parse_args()

    if args.config:
        run_from_config(load_config(args.config))
    else:
        _legacy_main()
