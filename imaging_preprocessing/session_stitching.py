"""Reusable helpers for fixing / stitching imaging-session recordings.

This file used to be a scratchpad of one-off corrections (specific mice, hard
coded server paths). Those operations actually reuse a small set of verbs, which
are now exposed as documented functions so any session can be fixed without
editing code:

  * reading ``log_continuous.bin`` and pulling out its interleaved channels,
  * detecting imaging-frame / trial / camera events,
  * truncating a log at a sample (e.g. an accidental extra recording),
  * stitching two split sessions (logs + behaviour tables),
  * counting / truncating / merging imaging tiffs and behaviour-camera avis.

``cv2`` and ``ScanImageTiffReader`` are imported lazily inside the functions that
need them, so importing this module never requires them.

Channel layout of log_continuous.bin
-------------------------------------
The binary is a flat float64 array of ``N_LOG_CHANNELS`` interleaved analog
channels sampled at ``LOG_SAMPLING_RATE`` Hz. Channel ``i`` is ``log[i::6]``.
The defaults below match the LSENS rig; override via ``LOG_CHANNELS`` if your
wiring differs.

Example: stitch two split sessions
----------------------------------
    log1 = read_log('.../sess1/log_continuous.bin')
    log2 = read_log('.../sess2/log_continuous.bin')
    merged = stitch_logs(log1, log2)            # optionally trim_end1=12000
    write_log(merged, '.../corrected/log_continuous.bin')

    import pandas as pd
    df1 = pd.read_csv('.../sess1/results.csv')
    df2 = pd.read_csv('.../sess2/results.csv')
    full = stitch_behavior(df1, df2, part1_duration_s=log_duration_s(log1))
    full.to_csv('.../corrected/results.csv', index=False)

Example: cut an accidental over-recording at the last trial
-----------------------------------------------------------
    log = read_log(path)
    starts = detect_trial_starts(log)
    cut = starts[-1] + 6 * LOG_SAMPLING_RATE     # 6 s after last trial start
    write_log(truncate_log(log, cut), out_path)
"""

import argparse
import os

import numpy as np
from scipy.signal import find_peaks


# --- Log format constants --------------------------------------------------
LOG_SAMPLING_RATE = 5000   # Hz
N_LOG_CHANNELS = 6

# Logical channel name -> interleave index into the flat log array.
LOG_CHANNELS = {
    'galvo': 1,     # imaging-frame (galvo) pulses -> one peak per imaging frame
    'trial': 2,     # trial-start TTL (thresholded > 2 V)
    'camera': 3,    # behaviour-camera frame pulses
}


# --- Reading / writing logs ------------------------------------------------
def read_log(path):
    """Read a ``log_continuous.bin`` file into a flat float64 array."""
    return np.fromfile(path)


def write_log(log, path):
    """Write a log array back to a binary file (creates parent dirs)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode='wb') as fid:
        log.tofile(fid)


def log_channel(log, channel, n_channels=N_LOG_CHANNELS):
    """Return one de-interleaved channel.

    ``channel`` may be an integer index or a key of :data:`LOG_CHANNELS`.
    """
    idx = LOG_CHANNELS[channel] if isinstance(channel, str) else channel
    return log[idx::n_channels]


def log_duration_s(log, fs=LOG_SAMPLING_RATE, n_channels=N_LOG_CHANNELS):
    """Duration of a log in seconds."""
    return np.round(len(log) / n_channels) / fs


# --- Event detection -------------------------------------------------------
def detect_galvo_frames(log, distance=100, prominence=1):
    """Sample indices of imaging-frame (galvo) pulses."""
    return find_peaks(log_channel(log, 'galvo'),
                      distance=distance, prominence=prominence)[0]


def detect_camera_frames(log, distance=10, prominence=1):
    """Sample indices of behaviour-camera frame pulses."""
    return find_peaks(log_channel(log, 'camera'),
                      distance=distance, prominence=prominence)[0]


def detect_trial_starts(log, threshold=2.0, distance=100, prominence=1):
    """Sample indices of trial-start TTL rising edges."""
    ttl = log_channel(log, 'trial')
    rising = (ttl > threshold)[1:].astype(np.float64) - (ttl > threshold)[:-1].astype(np.float64)
    return find_peaks(rising, distance=distance, prominence=prominence)[0]


# --- Log editing -----------------------------------------------------------
def truncate_log(log, cut_sample, n_channels=N_LOG_CHANNELS):
    """Return a copy of ``log`` with every channel zeroed after ``cut_sample``.

    ``cut_sample`` is an index into a single de-interleaved channel.
    """
    out = np.copy(log)
    for i in range(n_channels):
        out[i::n_channels][cut_sample:] = 0
    return out


def stitch_logs(log1, log2, trim_end1=0):
    """Concatenate two logs, optionally dropping ``trim_end1`` samples (per
    channel) from the end of the first (e.g. a stray TTL from stopping)."""
    if trim_end1:
        log1 = log1[:-trim_end1 * N_LOG_CHANNELS]
    return np.concatenate([log1, log2])


# --- Behaviour tables ------------------------------------------------------
def stitch_behavior(df1, df2, part1_duration_s):
    """Concatenate two behaviour tables, offsetting the second's trial time and
    trial number so they continue after the first session."""
    df2 = df2.copy()
    n_trials_part1 = df1['trial_number'].max()
    df2['trial_time'] = df2['trial_time'] + part1_duration_s
    df2['trial_number'] = df2['trial_number'] + n_trials_part1
    import pandas as pd
    return pd.concat([df1, df2]).reset_index(drop=True)


# --- Frame counting --------------------------------------------------------
def count_tiff_frames(path):
    """Number of frames in a single ScanImage tiff."""
    from ScanImageTiffReader import ScanImageTiffReader
    return ScanImageTiffReader(path).shape()[0]


def count_tiff_frames_in_folder(folder, extensions=('.tif', '.tiff')):
    """Total frames across all tiffs in a folder."""
    total = 0
    for name in os.listdir(folder):
        if name.lower().endswith(extensions):
            total += count_tiff_frames(os.path.join(folder, name))
    return total


def count_avi_frames(path):
    """Number of frames reported by a behaviour-camera avi."""
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# --- Tiff / avi editing ----------------------------------------------------
def truncate_tiff(in_path, out_path, drop_last):
    """Write a copy of a tiff with the last ``drop_last`` frames removed."""
    import tifffile
    from ScanImageTiffReader import ScanImageTiffReader
    data = ScanImageTiffReader(in_path).data()
    if drop_last:
        data = data[:-drop_last]
    tifffile.imwrite(out_path, data)


def truncate_avi(in_path, out_path, keep_frames, fps=100.0, size=(640, 480),
                 fourcc='Y800'):
    """Re-write an avi keeping only the first ``keep_frames`` frames."""
    import cv2
    cap = cv2.VideoCapture(in_path)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
    count = 0
    while cap.isOpened() and count < keep_frames:
        ret, frame = cap.read()
        if frame is None:
            break
        writer.write(frame)
        count += 1
    cap.release()
    writer.release()
    return count


def merge_avis(in_paths, out_path, fps=100.0, size=(640, 480), fourcc='Y800'):
    """Concatenate several avis into one (e.g. a split filming session)."""
    import cv2
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
    n = 0
    for path in in_paths:
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            writer.write(frame)
            n += 1
        cap.release()
    writer.release()
    return n


# --- Thin CLI for the most common operations -------------------------------
def _cli_stitch_logs(args):
    log1, log2 = read_log(args.log1), read_log(args.log2)
    merged = stitch_logs(log1, log2, trim_end1=args.trim_end1)
    write_log(merged, args.out)
    print(f'Wrote {args.out} ({log_duration_s(merged):.1f} s).')


def _cli_truncate_log(args):
    log = read_log(args.log)
    if args.at_last_trial:
        cut = detect_trial_starts(log)[-1] + args.post_trial_s * LOG_SAMPLING_RATE
    else:
        cut = args.cut_sample
    write_log(truncate_log(log, int(cut)), args.out)
    print(f'Wrote {args.out}, cut at sample {int(cut)}.')


def _cli_count_frames(args):
    if args.log:
        print('galvo (imaging) frames :', detect_galvo_frames(read_log(args.log)).size)
        print('camera frames          :', detect_camera_frames(read_log(args.log)).size)
    if args.tiff:
        print('tiff frames            :', count_tiff_frames(args.tiff))
    if args.tiff_folder:
        print('tiff frames (folder)   :', count_tiff_frames_in_folder(args.tiff_folder))
    if args.avi:
        print('avi frames             :', count_avi_frames(args.avi))


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('stitch-logs', help='Concatenate two split logs.')
    p.add_argument('log1'); p.add_argument('log2'); p.add_argument('out')
    p.add_argument('--trim-end1', type=int, default=0,
                   help='Samples (per channel) to drop from end of log1.')
    p.set_defaults(func=_cli_stitch_logs)

    p = sub.add_parser('truncate-log', help='Zero a log after a sample.')
    p.add_argument('log'); p.add_argument('out')
    p.add_argument('--cut-sample', type=int, default=None)
    p.add_argument('--at-last-trial', action='store_true',
                   help='Cut a fixed time after the last detected trial start.')
    p.add_argument('--post-trial-s', type=float, default=6.0)
    p.set_defaults(func=_cli_truncate_log)

    p = sub.add_parser('count-frames', help='Count frames in a log/tiff/avi.')
    p.add_argument('--log'); p.add_argument('--tiff')
    p.add_argument('--tiff-folder'); p.add_argument('--avi')
    p.set_defaults(func=_cli_count_frames)

    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    args.func(args)
