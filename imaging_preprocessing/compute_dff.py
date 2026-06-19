import argparse
import os

import scipy
import numpy as np
from tqdm import tqdm
from natsort import natsorted

from config import (
    EXPERIMENTER_MAP,
    get_experimenter_analysis_folder,
    load_config,
)


def set_merged_roi_to_non_cell(stat, iscell):
    # Set merged cells to 0 in iscell.
    if 'inmerge' in stat[0].keys():
        print('Cells in merge')
        for i, st in enumerate(stat):
            # 0: no merge; -1: input of a merge; index > 0: result of a merge.
            if st['inmerge'] not in [0, -1]:
                iscell[i][0] = 0.0

    return iscell


def compute_baseline(F, fs, window, sigma_win=5):

    # Parameters --------------------------------------------------------------
    nfilt = 30  # Number of taps to use in FIR filter
    fw_base = 1  # Cut-off frequency for lowpass filter, in Hz
    base_pctle = 5  # Percentile to take as baseline value

    # Main --------------------------------------------------------------------
    # Ensure array_like input is a numpy.ndarray
    F = np.asarray(F)

    # For short measurements, we reduce the number of taps
    nfilt = min(nfilt, max(3, int(F.shape[1] / 3)))

    if fs <= fw_base:
        # If our sampling frequency is less than our goal with the smoothing
        # (sampling at less than 1Hz) we don't need to apply the filter.
        filtered_f = F
    else:
        # The Nyquist rate of the signal is half the sampling frequency
        nyq_rate = fs / 2.0

        # Cut-off needs to be relative to the nyquist rate. For sampling
        # frequencies in the range from our target lowpass filter, to
        # twice our target (i.e. the 1Hz to 2Hz range) we instead filter
        # at the Nyquist rate, which is the highest possible frequency to
        # filter at.
        cutoff = min(1.0, fw_base / nyq_rate)

        # Make a set of weights to use with our taps.
        # We use an FIR filter with a Hamming window.
        b = scipy.signal.firwin(nfilt, cutoff=cutoff, window='hamming')

        # The default padlen for filtfilt is 3 * nfilt, but in case our
        # dataset is small, we need to make sure padlen is not too big
        padlen = min(3 * nfilt, F.shape[1] - 1)

        # Use filtfilt to filter with the FIR filter, both forwards and
        # backwards.
        filtered_f = scipy.signal.filtfilt(b, [1.0], F, axis=1,
                                           padlen=padlen)

    # Take a percentile of the filtered signal and windowed signal
    # baseline = scipy.ndimage.percentile_filter(filtered_f, percentile=base_pctle, size=(1,int(np.round(fs*2*window + 1))), mode='constant', cval=+np.inf)
    baseline = scipy.ndimage.minimum_filter(filtered_f, size=(1,int(np.round(fs*2*window + 1))), mode='reflect')
    baseline = scipy.ndimage.maximum_filter(baseline, size=(1,int(np.round(fs*2*window + 1))), mode='reflect')

    # Smooth baseline with gaussian filter.
    baseline = scipy.ndimage.gaussian_filter(baseline, sigma=(0, int(np.round(fs*sigma_win)) ), mode='reflect')

    # Ensure filtering doesn't take us below the minimum value which actually
    # occurs in the data. This can occur when the amount of data is very low.
    baseline = np.maximum(baseline, np.nanmin(F, axis=1, keepdims=True))

    return baseline, filtered_f


def compute_dff(F_raw, F_neu, fs, window=30, sigma_win=5, neuropil_coeff=0.7):
    '''
    F_cor: decontaminated traces, output of Fissa
    F_raw: raw traces extracted by Fissa (not suite2p)
    fs: sampling frequency
    window: running window size on each side of sample for percentile computation
    neuropil_coeff: neuropil contamination coefficient subtracted from F_raw
    '''
    F_cor = F_raw - neuropil_coeff * F_neu
    F_cor[F_cor<0] = 0  # Ensure non negative values.
    F0_raw, _ = compute_baseline(F_raw, fs, window, sigma_win=sigma_win)
    F0_raw[F0_raw<1] = 1  # Avoid division by < 1.
    F0_cor, _ = compute_baseline(F_cor, fs, window, sigma_win=sigma_win)
    dff = (F_cor - F0_cor) / F0_raw

    return F0_raw, F0_cor, dff  


# EXPERIMENTER_MAP and get_experimenter_analysis_folder now live in config.py
# (shared with run_suite2p) and are imported at the top of this file.


def find_suite2p_folders(experimenter, mice_ids, haas_path=False,
                         suite2p_subpath=('suite2p', 'plane0'),
                         overwrite=False, paths=None, experimenter_map=None):
    """Collect the Suite2p output folders to process for the given mice.

    Skips folders that already have a ``dff.npy`` unless ``overwrite`` is set.
    """
    analysis_root = get_experimenter_analysis_folder(
        experimenter, haas_path, paths, experimenter_map)
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
            if not overwrite and os.path.exists(os.path.join(suite2p_folder, 'dff.npy')):
                continue
            suite2p_folders.append(suite2p_folder)
    return suite2p_folders


def process_suite2p_folder(suite2p_folder, window=30, sigma_win=5,
                           neuropil_coeff=0.7):
    """Compute and save F_raw/F_neu/F0/dff for a single Suite2p output folder."""
    if not os.path.exists(os.path.join(suite2p_folder, 'stat.npy')):
        return

    tqdm.write(f'\nProcessing {suite2p_folder}.')
    stat = np.load(os.path.join(suite2p_folder, 'stat.npy'), allow_pickle=True)
    ops = np.load(os.path.join(suite2p_folder, 'ops.npy'), allow_pickle=True).item()
    iscell = np.load(os.path.join(suite2p_folder, 'iscell.npy'), allow_pickle=True)
    F_raw = np.load(os.path.join(suite2p_folder, 'F.npy'), allow_pickle=True)
    F_neu = np.load(os.path.join(suite2p_folder, 'Fneu.npy'), allow_pickle=True)

    # Set merged roi's to non-cells.
    iscell = set_merged_roi_to_non_cell(stat, iscell)

    F_raw = F_raw[iscell[:, 0] == 1.]
    F_neu = F_neu[iscell[:, 0] == 1.]

    tqdm.write('Computing baselines and dff.')
    F0_raw, F0_cor, dff = compute_dff(F_raw, F_neu, fs=ops['fs'], window=window,
                                      sigma_win=sigma_win,
                                      neuropil_coeff=neuropil_coeff)

    # Saving data.
    np.save(os.path.join(suite2p_folder, 'F_raw'), F_raw)
    np.save(os.path.join(suite2p_folder, 'F_neu'), F_neu)
    np.save(os.path.join(suite2p_folder, 'F0_cor'), F0_cor)
    np.save(os.path.join(suite2p_folder, 'F0_raw'), F0_raw)
    np.save(os.path.join(suite2p_folder, 'dff'), dff)
    tqdm.write(f'Data saved : {suite2p_folder}')


def run_from_config(config):
    """Compute dff for all requested mice/sessions from a YAML config dict."""
    dff_cfg = config.get('dff') or {}
    layout = config.get('folder_layout') or {}

    suite2p_folders = find_suite2p_folders(
        experimenter=config['experimenter'],
        mice_ids=config['mice'],
        haas_path=config.get('on_haas', False),
        suite2p_subpath=tuple(layout.get('suite2p_subpath', ('suite2p', 'plane0'))),
        overwrite=dff_cfg.get('overwrite', False),
        paths=config.get('paths'),
        experimenter_map=config.get('experimenter_map'),
    )
    print(suite2p_folders)
    for suite2p_folder in tqdm(suite2p_folders, desc='Processing suite2p folders'):
        process_suite2p_folder(
            suite2p_folder,
            window=dff_cfg.get('window', 30),
            sigma_win=dff_cfg.get('sigma_win', 5),
            neuropil_coeff=dff_cfg.get('neuropil_coeff', 0.7),
        )


def _legacy_main():
    """The original hardcoded run, kept so existing usage still works."""
    suite2p_folders = find_suite2p_folders('RD', ['RD119', 'RD121'], haas_path=True)
    print(suite2p_folders)
    for suite2p_folder in tqdm(suite2p_folders, desc='Processing suite2p folders'):
        process_suite2p_folder(suite2p_folder, window=30, sigma_win=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute dF/F for Suite2p outputs from a YAML config.')
    parser.add_argument('--config', '-c', default=None,
                        help='Path to a YAML run config. If omitted, runs the '
                             'legacy hardcoded settings.')
    args = parser.parse_args()

    if args.config:
        run_from_config(load_config(args.config))
    else:
        _legacy_main()



