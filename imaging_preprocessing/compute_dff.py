import os
import scipy
import numpy as np
from tqdm import tqdm
from natsort import natsorted


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


def compute_dff(F_raw, F_neu, fs, window=30, sigma_win=5):
    '''
    F_cor: decontaminated traces, output of Fissa
    F_raw: raw traces extracted by Fissa (not suite2p)
    fs: sampling frequency
    window: running window size on each side of sample for percentile computation
    '''
    F_cor = F_raw - 0.7 * F_neu
    F_cor[F_cor<0] = 0  # Ensure non negative values.
    F0_raw, _ = compute_baseline(F_raw, fs, window, sigma_win=5)
    F0_raw[F0_raw<1] = 1  # Avoid division by < 1.
    F0_cor, _ = compute_baseline(F_cor, fs, window, sigma_win=5)
    dff = (F_cor - F0_cor) / F0_raw

    return F0_raw, F0_cor, dff  


EXPERIMENTER_MAP = {
    'AR': 'Anthony_Renard',
    'RD': 'Robin_Dard',
    'AB': 'Axel_Bisi',
    'MP': 'Mauro_Pulin',
    'PB': 'Pol_Bech',
    'MM': 'Meriam_Malekzadeh',
    'MS': 'Lana_Smith',
    'GF': 'Anthony_Renard',
    'MI': 'Anthony_Renard',
    'AS': 'Morgane_Storey',
}


def get_experimenter_analysis_folder(initials, haas_path=False):
    # Map initials to experimenter to get analysis folder path.
    experimenter = EXPERIMENTER_MAP[initials]
    if not haas_path:
        analysis_folder = os.path.join('//sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis',
                                       experimenter, 'data')
    else:
        analysis_folder = os.path.join('/mnt', 'lsens-analysis',
                                       experimenter, 'data')
    return analysis_folder


# Compute dff for full data (sessions not split).
# -------------------------------------------------

experimenter = 'RD'
mice_ids = ['RD119', 'RD121']
on_haas = True
suite2p_folders = []
for mouse_id in mice_ids:
    session_ids = natsorted(os.listdir(os.path.join(get_experimenter_analysis_folder(experimenter, on_haas), mouse_id)))
    for session_id in session_ids:
        session_folder = os.path.join(get_experimenter_analysis_folder(experimenter, on_haas), mouse_id, session_id)
        if not os.path.isdir(session_folder):
            continue
        suite2p_folder = os.path.join(session_folder, 'suite2p', 'plane0')
        if not os.path.isdir(suite2p_folder):
            continue
        dff_file = os.path.join(suite2p_folder, 'dff.npy')
        if os.path.exists(dff_file):
            continue
        suite2p_folders.append(suite2p_folder)

print(suite2p_folders)
for suite2p_folder in tqdm(suite2p_folders, desc='Processing suite2p folders'):

    if not os.path.exists(os.path.join(suite2p_folder,'stat.npy')):
        continue

    tqdm.write(f'\nProcessing {suite2p_folder}.')
    stat = np.load(os.path.join(suite2p_folder,'stat.npy'), allow_pickle = True)
    ops = np.load(os.path.join(suite2p_folder,'ops.npy'), allow_pickle = True).item()
    iscell = np.load(os.path.join(suite2p_folder,'iscell.npy'), allow_pickle = True)
    F_raw = np.load(os.path.join(suite2p_folder,'F.npy'), allow_pickle = True)
    F_neu = np.load(os.path.join(suite2p_folder,'Fneu.npy'), allow_pickle = True)
    
    # Set merged roi's to non-cells.
    iscell = set_merged_roi_to_non_cell(stat, iscell)

    F_raw = F_raw[iscell[:,0]==1.]
    F_neu = F_neu[iscell[:,0]==1.]

    tqdm.write('Computing baselines and dff.')
    F0_raw, F0_cor, dff = compute_dff(F_raw, F_neu, fs=ops['fs'], window=30, sigma_win=5)

    # Saving data.
    np.save(os.path.join(suite2p_folder, 'F_raw'), F_raw)
    np.save(os.path.join(suite2p_folder, 'F_neu'), F_neu)
    np.save(os.path.join(suite2p_folder, 'F0_cor'), F0_cor)
    np.save(os.path.join(suite2p_folder, 'F0_raw'), F0_raw)
    np.save(os.path.join(suite2p_folder, 'dff'), dff)
    tqdm.write(f'Data saved : {suite2p_folder}')



