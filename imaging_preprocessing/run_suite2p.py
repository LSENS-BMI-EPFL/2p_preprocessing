import os
import sys
import logging
from datetime import datetime
from natsort import natsorted

from suite2p import run_s2p, default_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
sys.stdout.reconfigure(line_buffering=True)


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
    }


def get_data_folder(haas_path=False):
    if not haas_path:
        data_folder = os.path.join('//sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'data')
    else:
        data_folder = os.path.join('/mnt', 'lsens-data')

    return data_folder  


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


def run(ops, mice_ids, experimenter, longitudinal=False,
        excluded_sessions=None, haas_path=False, date=None, session_to_do=None):
    dbs = []
    if longitudinal:
        # Concatenate all sessions and run suite2p once per mouse.
        for mouse_id in mice_ids:
            tiff_folders = os.path.join(get_data_folder(haas_path), mouse_id, 'Recording', 'Imaging')
            tiff_folders = [os.path.join(tiff_folders, folder) for folder in os.listdir(tiff_folders)
                            if os.path.isdir(os.path.join(tiff_folders, folder))]
            tiff_folders = natsorted(tiff_folders)            
            
            save_path = os.path.join(get_experimenter_analysis_folder(experimenter, haas_path), mouse_id)
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            # db overwrites any ops (allows for experiment specific settings)
            db = {
                'h5py': [], # a single h5 file path
                'h5py_key': 'data',
                'data_path': tiff_folders, # a list of folders with tiffs
                                                        # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
                'save_path0': save_path,
                }
            dbs.append(db)
        print(f'dbs : {dbs}', flush=True)
    else:
        # Run suite2p for each session.
        for mouse_id in mice_ids:
            tiff_folders = os.path.join(get_data_folder(haas_path), mouse_id, 'Recording', 'Imaging')
            tiff_folders = [os.path.join(tiff_folders, folder)
                            for folder in os.listdir(tiff_folders)
                            if os.path.isdir(os.path.join(tiff_folders, folder))]
            tiff_folders = natsorted(tiff_folders)

            for folder in tiff_folders:
                session_id = os.path.split(folder)[1]
                session_date = datetime.strptime(session_id.split('_')[1], '%Y%m%d')

                # Exclusion criteria
                if date:
                    if session_date < datetime.strptime(date, '%Y%m%d'):
                        continue
                if excluded_sessions and session_id in excluded_sessions:
                    continue
                if session_to_do and session_id != session_to_do:
                    continue

                print(f'session_id :{session_id}', flush=True)
                save_path = os.path.join(get_experimenter_analysis_folder(experimenter, haas_path),
                                         mouse_id, session_id)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # db overwrites any ops (allows for experiment specific settings)
                db = {
                    'h5py': [], # a single h5 file path
                    'h5py_key': 'data',
                    'data_path': [folder], # a list of folders with tiffs
                    # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
                    'save_path0': save_path,
                    }
                dbs.append(db)
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


if __name__ == '__main__':
    
    mice = ['RD116', 'RD119', 'RD121']
    # mice = ['RD119']
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
    opts = default_settings() # populates opts with the default options

    opts['fs'] = 30
    opts['tau'] = 0.1  # timescale of gcamp8s to use for deconvolution

    opts['io']['delete_bin'] = False

    opts['registration']['batch_size'] = 500
    opts['registration']['nimg_init'] = 400

    opts['registration']['reg_tif'] = True  # Save reg tif for Fissa and NWB

    opts['detection']['threshold_scaling'] = 0.9
    opts['detection']['sparsery_settings']['max_ROIs'] = 1000


    run(opts, mice, experimenter_ID, longitudinal_ci, excluded_sess, on_haas, start_date, session_id_to_do)
