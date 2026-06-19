import argparse
import os
import sys
import logging
from datetime import datetime

from natsort import natsorted

from suite2p import run_s2p, default_settings

from config import (
    EXPERIMENTER_MAP,
    DEFAULT_PATHS,
    get_data_folder,
    get_experimenter_analysis_folder,
    deep_update,
    load_config,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
sys.stdout.reconfigure(line_buffering=True)


# EXPERIMENTER_MAP, get_data_folder and get_experimenter_analysis_folder now
# live in config.py so run_suite2p and compute_dff share a single source of
# truth. They are re-imported above for backward compatibility.


def build_session_dbs(mice_ids, experimenter, longitudinal=False,
                      excluded_sessions=None, haas_path=False, date=None,
                      session_to_do=None, imaging_subpath=('Recording', 'Imaging'),
                      session_date_format='%Y%m%d', paths=None,
                      experimenter_map=None):
    """Build the list of Suite2p ``db`` dicts for the requested sessions.

    Folder layout and date parsing are parameters now (with the lab defaults)
    so a different naming convention only needs a config change, not a code
    edit.
    """
    paths = paths or DEFAULT_PATHS
    experimenter_map = experimenter_map or EXPERIMENTER_MAP
    dbs = []

    for mouse_id in mice_ids:
        tiff_root = os.path.join(get_data_folder(haas_path, paths), mouse_id,
                                 *imaging_subpath)
        tiff_folders = [os.path.join(tiff_root, folder)
                        for folder in os.listdir(tiff_root)
                        if os.path.isdir(os.path.join(tiff_root, folder))]
        tiff_folders = natsorted(tiff_folders)

        analysis_folder = get_experimenter_analysis_folder(
            experimenter, haas_path, paths, experimenter_map)

        if longitudinal:
            # Concatenate all sessions and run suite2p once per mouse.
            save_path = os.path.join(analysis_folder, mouse_id)
            os.makedirs(save_path, exist_ok=True)
            dbs.append({
                'h5py': [],
                'h5py_key': 'data',
                'data_path': tiff_folders,  # a list of folders with tiffs
                'save_path0': save_path,
            })
        else:
            # Run suite2p for each session.
            for folder in tiff_folders:
                session_id = os.path.split(folder)[1]
                session_date = datetime.strptime(
                    session_id.split('_')[1], session_date_format)

                # Exclusion criteria.
                if date and session_date < datetime.strptime(date, session_date_format):
                    continue
                if excluded_sessions and session_id in excluded_sessions:
                    continue
                if session_to_do and session_id != session_to_do:
                    continue

                print(f'session_id :{session_id}', flush=True)
                save_path = os.path.join(analysis_folder, mouse_id, session_id)
                os.makedirs(save_path, exist_ok=True)
                dbs.append({
                    'h5py': [],
                    'h5py_key': 'data',
                    'data_path': [folder],  # a list of folders with tiffs
                    'save_path0': save_path,
                })

    return dbs


def run(ops, mice_ids, experimenter, longitudinal=False,
        excluded_sessions=None, haas_path=False, date=None, session_to_do=None,
        imaging_subpath=('Recording', 'Imaging'), session_date_format='%Y%m%d',
        paths=None, experimenter_map=None):
    """Run Suite2p over the requested mice/sessions.

    Signature is backward compatible: the original positional arguments behave
    exactly as before; the new keyword arguments expose folder layout so it can
    be driven from a config file.
    """
    dbs = build_session_dbs(
        mice_ids, experimenter, longitudinal, excluded_sessions, haas_path,
        date, session_to_do, imaging_subpath, session_date_format, paths,
        experimenter_map)
    print(f'dbs : {dbs}', flush=True)

    n = len(dbs)
    for i, dbi in enumerate(dbs):
        print(f'\n=== [{i+1}/{n}] Starting suite2p on: {dbi["data_path"]} ===', flush=True)
        try:
            run_s2p(db=dbi, settings=ops)
        except Exception as e:
            import traceback
            print(f'ERROR in run_s2p: {e}', flush=True)
            traceback.print_exc()
        print(f'=== [{i+1}/{n}] Done ===\n', flush=True)


def run_from_config(config):
    """Run the pipeline from a loaded YAML config dict (see configs/)."""
    # Start from Suite2p defaults and layer the user's overrides on top so the
    # config only has to list the options it changes.
    ops = deep_update(default_settings(), config.get('suite2p_ops'))

    layout = config.get('folder_layout') or {}
    run(
        ops=ops,
        mice_ids=config['mice'],
        experimenter=config['experimenter'],
        longitudinal=config.get('longitudinal', False),
        excluded_sessions=config.get('excluded_sessions'),
        haas_path=config.get('on_haas', False),
        date=config.get('start_date'),
        session_to_do=config.get('session_to_do'),
        imaging_subpath=tuple(layout.get('imaging_subpath', ('Recording', 'Imaging'))),
        session_date_format=layout.get('session_date_format', '%Y%m%d'),
        paths=config.get('paths'),
        experimenter_map=config.get('experimenter_map'),
    )


def _legacy_main():
    """The original hardcoded run, kept so existing usage still works."""
    mice = ['RD116', 'RD119', 'RD121']
    experimenter_ID = 'RD'
    longitudinal_ci = False
    on_haas = True
    excluded_sess = [
        'RD121_20260428_115902',
        'RD116_20260507_094904',
    ]
    start_date = '20260610'
    session_id_to_do = None

    # set your options for running
    opts = default_settings()  # populates opts with the default options
    opts['fs'] = 30
    opts['tau'] = 0.1  # timescale of gcamp8s to use for deconvolution
    opts['io']['delete_bin'] = False
    opts['registration']['batch_size'] = 500
    opts['registration']['nimg_init'] = 400
    opts['registration']['reg_tif'] = True  # Save reg tif for Fissa and NWB
    opts['detection']['threshold_scaling'] = 0.9
    opts['detection']['sparsery_settings']['max_ROIs'] = 1000

    run(opts, mice, experimenter_ID, longitudinal_ci, excluded_sess, on_haas,
        start_date, session_id_to_do)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Suite2p over mice/sessions from a YAML config.')
    parser.add_argument('--config', '-c', default=None,
                        help='Path to a YAML run config. If omitted, runs the '
                             'legacy hardcoded settings.')
    args = parser.parse_args()

    if args.config:
        run_from_config(load_config(args.config))
    else:
        _legacy_main()
