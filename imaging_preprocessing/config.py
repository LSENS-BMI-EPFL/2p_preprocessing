"""Shared configuration helpers for the 2P preprocessing pipeline.

This module centralises everything that used to be copy-pasted (and drift)
between the individual scripts:

  * the experimenter-initials -> folder-name map,
  * the data / analysis folder roots (lab server vs HAAS cluster mount),
  * loading run parameters from a YAML file instead of editing ``__main__``.

Nothing here is specific to a single run; per-run choices (which mice, which
dates, which Suite2p options) live in a YAML config file that gets loaded with
:func:`load_config`.
"""

import copy
import os

import yaml


# Map experimenter initials to the analysis sub-folder name on the server.
# Defined ONCE here; both run_suite2p and compute_dff import it so the two
# can no longer drift apart.
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


# Default storage roots. A config file may override either of these via the
# ``paths:`` block, e.g. to point at a different lab server or a local copy.
DEFAULT_PATHS = {
    # Raw data (tiffs) live here.
    'data_root': os.path.join('//sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'data'),
    'data_root_haas': os.path.join('/mnt', 'lsens-data'),
    # Suite2p output / analysis lives here, under <experimenter>/data.
    'analysis_root': os.path.join('//sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis'),
    'analysis_root_haas': os.path.join('/mnt', 'lsens-analysis'),
}


def get_data_folder(haas_path=False, paths=None):
    """Root folder containing the raw imaging data, per mouse."""
    paths = paths or DEFAULT_PATHS
    return paths['data_root_haas'] if haas_path else paths['data_root']


def get_experimenter_analysis_folder(initials, haas_path=False, paths=None,
                                     experimenter_map=None):
    """Analysis folder for an experimenter, e.g. <root>/Robin_Dard/data."""
    paths = paths or DEFAULT_PATHS
    experimenter_map = experimenter_map or EXPERIMENTER_MAP
    if initials not in experimenter_map:
        raise KeyError(
            f"Unknown experimenter initials {initials!r}. Known: "
            f"{sorted(experimenter_map)}. Add it to EXPERIMENTER_MAP in "
            f"config.py or to the config's experimenter_map block.")
    experimenter = experimenter_map[initials]
    root = paths['analysis_root_haas'] if haas_path else paths['analysis_root']
    return os.path.join(root, experimenter, 'data')


def deep_update(base, overrides):
    """Recursively merge ``overrides`` into ``base`` (returns a new dict).

    Used to layer user-supplied Suite2p options on top of the defaults without
    discarding the keys the user didn't mention.
    """
    result = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path):
    """Load a YAML config file into a plain dict.

    The returned dict is passed straight to the ``run_from_config`` entry point
    of each script. Resolves ``paths`` and ``experimenter_map`` against the
    defaults so callers always get a complete set.
    """
    with open(path, 'r') as fh:
        config = yaml.safe_load(fh) or {}

    # Merge any path / map overrides on top of the package defaults so the rest
    # of the code can assume the keys are present.
    config['paths'] = deep_update(DEFAULT_PATHS, config.get('paths'))
    config['experimenter_map'] = deep_update(EXPERIMENTER_MAP,
                                             config.get('experimenter_map'))
    return config
