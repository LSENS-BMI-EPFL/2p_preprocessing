import argparse
import os
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
    Turn suite2p badframes into artifact periods.
    suite2p only flags scattered frames inside a motion artifact, so flagged
    frames closer than merge_gap (s) are merged into one artifact, and each
    artifact is extended by pad (s) on both sides.
    Returns the boolean frame mask and the number of artifacts.
    '''
    mask = np.asarray(badframes).astype(bool)
    if mask.any():
        gap = max(1, int(round(merge_gap * fs)))
        mask = scipy.ndimage.binary_closing(np.pad(mask, gap), structure=np.ones(gap + 1))[gap:-gap]
        mask = scipy.ndimage.binary_dilation(mask, structure=np.ones(2 * int(round(pad * fs)) + 1))
    _, n_artifacts = scipy.ndimage.label(mask)

    return mask, n_artifacts


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


def compute_noise_metrics(F_cor, dff, snr_percentile=99.9):
    '''
    Per-cell noise metrics.
    noise_abs: frame noise of F_cor (a.u.)
    noise_rel: frame noise of dff (dF/F0 units, i.e. noise relative to baseline)
    snr: high percentile of dff (largest transients) / noise_rel. 99.9 rather
         than 99 so it reflects transient amplitude more than how often the
         cell is active.
    '''
    noise_abs = frame_noise(F_cor)
    noise_rel = frame_noise(dff)
    snr = np.nanpercentile(dff, snr_percentile, axis=1) / noise_rel

    return pd.DataFrame({'noise_abs': noise_abs, 'noise_rel': noise_rel, 'snr': snr})


def compute_contamination_metrics(F_raw, F_neu, fs, neuropil_coef=0.7, bad_frames=None, band=(0.5, 30)):
    '''
    Per-cell neuropil contamination metrics.
    neuropil_ratio: neuropil_coef * median(Fneu) / median(F), share of the
                    baseline removed by neuropil subtraction (>= 1: F_cor <= 0).
    neuropil_f_rho: correlation of F and Fneu fluctuations, band-passed between
                   band[0] and band[1] seconds (removes frame noise and slow drift).
    contamination: neuropil_ratio * neuropil_f_rho. ~0: clean, > ~0.8: F is
                   mostly neuropil (likely not a usable cell).
    Bad frames are excluded.
    '''
    good = slice(None) if bad_frames is None else ~bad_frames
    ratio = neuropil_coef * np.median(F_neu[:, good], axis=1) / np.median(F_raw[:, good], axis=1)

    def bandpass(x):
        x = x.astype(np.float32)
        return (scipy.ndimage.gaussian_filter1d(x, band[0] * fs, axis=1)
                - scipy.ndimage.gaussian_filter1d(x, band[1] * fs, axis=1))[:, good]

    Fb, Nb = bandpass(F_raw), bandpass(F_neu)
    Fb -= Fb.mean(axis=1, keepdims=True)
    Nb -= Nb.mean(axis=1, keepdims=True)
    corr = (Fb * Nb).sum(axis=1) / np.sqrt((Fb ** 2).sum(axis=1) * (Nb ** 2).sum(axis=1))

    return pd.DataFrame({'neuropil_ratio': ratio, 'neuropil_f_rho': corr, 'contamination': ratio * corr})


def select_example_cells(metrics, mode='quality', n_cells=6):
    '''
    Cells shown in dff_check. Returns (cells, labels).
    mode='quality': n_cells // 3 good, median and bad cells. Quality score =
        mean of the percentile ranks of snr (high = good) and contamination
        (low = good). Ordered good -> median -> bad.
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
    order = np.argsort(-score)  # best first
    k = max(1, n_cells // 3)
    mid = n_total // 2
    good, median, bad = order[:k], order[mid - k // 2: mid - k // 2 + k], order[-k:]
    cells = np.concatenate([good, median, bad])
    labels = ['good'] * len(good) + ['median'] * len(median) + ['bad'] * len(bad)
    return cells, labels


def plot_noise_distributions(metrics, save_path, cells=None, contamination_threshold=0.8, neuropil_coef=0.7):
    '''
    Top: histograms of noise_abs, noise_rel, snr and contamination across cells.
    Bottom: neuropil ratio vs neuropil_f_rho colored by SNR, and SNR vs contamination.
    `cells` (indices) are marked so they can be matched with the dff_check figure.
    '''
    columns = [('noise_abs', 'absolute noise (a.u.)'),
               ('noise_rel', 'relative noise (dF/F0)'),
               ('snr', 'SNR (99.9th pct of dF/F0 / noise rel)'),
               ('contamination', 'contamination (neuropil ratio x corr)')]
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.4])
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax, (col, label) in zip(axes, columns):
        values = metrics[col].to_numpy()
        # Cells with a tiny F0 have extreme values: clip the x range at
        # median + 5 IQR so the bulk of the distribution stays readable.
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        xmax = min(values.max(), q50 + 5 * (q75 - q25))
        n_out = np.sum(values > xmax)
        ax.hist(values, bins=50, range=(values.min(), xmax), color='#2a78d6', edgecolor='#fcfcfb', linewidth=0.5)
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
        ax.set_xlabel(label)
        ax.set_ylabel('cells')
        ax.spines[['top', 'right']].set_visible(False)
    if cells is not None:
        axes[-1].plot([], [], color='#eb6834', lw=1, label='cells in dff_check')
        axes[-1].plot([], [], color='#0b0b0b', lw=1.5, ls='--', label='median')
        axes[-1].legend(loc='center right', frameon=False, fontsize=8)
    axes[-1].axvline(contamination_threshold, color='#e34948', lw=1, ls=':')

    # Contamination scatters.
    ratio, corr = metrics.neuropil_ratio.to_numpy(), metrics.neuropil_f_rho.to_numpy()
    contamination, snr = metrics.contamination.to_numpy(), metrics.snr.to_numpy()
    high = contamination > contamination_threshold
    ax_rc = fig.add_subplot(gs[1, :2])
    ax_cs = fig.add_subplot(gs[1, 2:])

    sc = ax_rc.scatter(ratio, corr, c=np.log10(np.clip(snr, 1, None)), s=14, cmap='viridis')
    ax_rc.scatter(ratio[high], corr[high], s=50, facecolors='none', edgecolors='#e34948',
                  label=f'contamination > {contamination_threshold} ({high.sum()} cells)')
    fig.colorbar(sc, ax=ax_rc, label='log10 SNR')
    ax_rc.set_xlabel(f'neuropil ratio  {neuropil_coef:g}·Fneu / F (baseline)')
    ax_rc.set_ylabel('neuropil_f_rho  corr(F, Fneu), 0.5-30 s')

    ax_cs.scatter(contamination, snr, s=14, color='#2a78d6')
    ax_cs.set_yscale('log')
    ax_cs.axvline(contamination_threshold, color='#e34948', lw=1, ls=':')
    ax_cs.set_xlabel('contamination (neuropil ratio x corr)')
    ax_cs.set_ylabel('SNR')

    if cells is not None:
        for ax, (x, y) in [(ax_rc, (ratio, corr)), (ax_cs, (contamination, snr))]:
            ax.scatter(x[cells], y[cells], s=60, marker='D', facecolors='none', edgecolors='#eb6834', lw=1.5,
                       label='cells in dff_check')
            for icell in cells:
                ax.annotate(str(icell), (x[icell], y[icell]), xytext=(5, 5), textcoords='offset points',
                            fontsize=8, color='#0b0b0b')
    for ax in (ax_rc, ax_cs):
        ax.legend(frameon=False, fontsize=8, loc='upper left')
        ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_dff_check(F_raw, F_neu, F_cor, F0, dff, fs, save_path, cells, metrics=None, bad_frames=None,
                   duration=300, seed=None, cell_labels=None):
    '''
    Plot F, Fneu, F_cor, F0 (left) and dff (right) for the given cells
    over a random segment of `duration` seconds. If metrics is given, the
    SNR and contamination of each cell are written above its dff trace. Artifact
    periods (bad_frames) are shaded.
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
        labels, n = scipy.ndimage.label(bad_frames[sl])
        artifacts = [(t[s][0], t[s][-1]) for s in scipy.ndimage.find_objects(labels)]

    fig, axes = plt.subplots(len(cells), 2, figsize=(20, 2 * len(cells)), sharex=True, squeeze=False)
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
            ax_d.text(1.0, 1.0, f'SNR = {m.snr:.3g}   contamination = {m.contamination:.2f}',
                      transform=ax_d.transAxes, ha='right', va='bottom', fontsize=9, color='#0b0b0b')
        for ax in (ax_f, ax_d):
            ax.spines[['top', 'right']].set_visible(False)
    if artifacts:
        axes[0, 0].axvspan(np.nan, np.nan, color='#898781', alpha=0.25, lw=0, label='artifact')
    axes[0, 0].legend(loc='upper right', ncol=5, frameon=False, fontsize=8)
    axes[0, 0].set_title('F, Fneu, F_cor, F0', pad=14)
    axes[0, 1].set_title('dF/F0', pad=14)
    axes[-1, 0].set_xlabel('time (s)')
    axes[-1, 1].set_xlabel('time (s)')
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
                           artifact_merge_gap=2, artifact_pad=2, example_cells='quality', plot_duration=180,
                           contamination_threshold=0.8):
    '''
    Compute and save F_cor, F0, dff, artifact frames, noise metrics and QC
    figures for a single suite2p output folder.
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

    # Motion artifacts from suite2p badframes.
    log('Detecting artifacts from suite2p badframes.')
    badframes = np.asarray(ops.get('badframes', np.zeros(F_raw.shape[1], dtype=bool))).astype(bool)
    bad_frames, n_artifacts = get_artifact_mask(badframes, fs=ops['fs'], merge_gap=artifact_merge_gap, pad=artifact_pad)
    log(f'{n_artifacts} artifacts | suite2p bad frames: {badframes.sum()} ({100 * badframes.mean():.3f}%) '
        f'| masked frames (merged + padded): {bad_frames.sum()} ({100 * bad_frames.mean():.3f}%)')

    log('Computing F_cor, F0 (rolling percentile) and dff.')
    # _bad: artifact frames kept. Without suffix: artifact frames set to NaN.
    F_cor_bad, F0, dff_bad = compute_dff(F_raw, F_neu, fs=ops['fs'], neuropil_coef=neuropil_coef, window=window,
                                         percentile=percentile, sigma=sigma, min_f0=min_f0, bad_frames=bad_frames)
    F_cor = mask_frames(F_cor_bad, bad_frames)
    dff = mask_frames(dff_bad, bad_frames)

    # Cells whose neuropil-corrected baseline is tiny get huge dff values.
    low_f0 = np.median(F0, axis=1) < 0.1 * np.median(F_raw, axis=1)
    if low_f0.any():
        log(f'{low_f0.sum()} cells with F0 < 10% of raw F (dff unreliable): {np.where(low_f0)[0]}')

    # Saving data.
    log('Saving F_raw, F_neu, F_cor, F_cor_bad, F0, dff, dff_bad, artifact_frames.')
    np.save(os.path.join(suite2p_folder, 'F_raw'), F_raw)
    np.save(os.path.join(suite2p_folder, 'F_neu'), F_neu)
    np.save(os.path.join(suite2p_folder, 'F_cor'), F_cor)
    np.save(os.path.join(suite2p_folder, 'F_cor_bad'), F_cor_bad)
    np.save(os.path.join(suite2p_folder, 'F0'), F0)
    np.save(os.path.join(suite2p_folder, 'dff'), dff)
    np.save(os.path.join(suite2p_folder, 'dff_bad'), dff_bad)
    np.save(os.path.join(suite2p_folder, 'artifact_frames'), bad_frames)

    # Noise metrics, on artifact-free traces. roi = index in the full suite2p roi list.
    log('Computing noise and contamination metrics.')
    metrics = pd.concat([compute_noise_metrics(F_cor, dff),
                         compute_contamination_metrics(F_raw, F_neu, fs=ops['fs'], neuropil_coef=neuropil_coef,
                                                       bad_frames=bad_frames)], axis=1)
    metrics.insert(0, 'roi', np.where(iscell[:, 0] == 1.)[0])
    metrics.to_csv(os.path.join(suite2p_folder, 'noise_metrics.csv'), index_label='cell')
    # One (n_cells,) array per metric, same cell order as dff.
    for col in ['noise_abs', 'noise_rel', 'snr', 'neuropil_ratio', 'neuropil_f_rho', 'contamination']:
        np.save(os.path.join(suite2p_folder, col), metrics[col].to_numpy())
    log(f'Median noise abs = {metrics.noise_abs.median():.3g} a.u. | noise rel = {metrics.noise_rel.median():.3g} '
        f'| SNR = {metrics.snr.median():.3g} | contamination = {metrics.contamination.median():.2f} '
        f'({(metrics.contamination > contamination_threshold).sum()} cells > {contamination_threshold})')

    # Figures, same example cells in both.
    log('Plotting dff_check.png and noise_distributions.png.')
    cells, cell_labels = select_example_cells(metrics, mode=example_cells)
    plot_dff_check(F_raw, F_neu, F_cor_bad, F0, dff, fs=ops['fs'], cells=cells, cell_labels=cell_labels,
                   duration=plot_duration, metrics=metrics,
                   bad_frames=bad_frames, save_path=os.path.join(suite2p_folder, 'dff_check.png'))
    plot_noise_distributions(metrics, cells=cells, contamination_threshold=contamination_threshold,
                             neuropil_coef=neuropil_coef,
                             save_path=os.path.join(suite2p_folder, 'noise_distributions.png'))
    log(f'Done. Data saved: {suite2p_folder}')


# Keys of the config's dff_updated: block passed to process_suite2p_folder.
DFF_PARAMS = ['neuropil_coef', 'window', 'percentile', 'sigma', 'min_f0', 'artifact_merge_gap',
              'artifact_pad', 'example_cells', 'plot_duration', 'contamination_threshold']


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
