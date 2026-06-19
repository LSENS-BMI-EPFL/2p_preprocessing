# 2p_data_preprocessing

Preprocessing pipeline for 2-photon calcium imaging data (Suite2p extraction +
ΔF/F computation).

## Configuration

Run parameters live in a **YAML config file** instead of being hardcoded in the
scripts. Copy [`imaging_preprocessing/configs/example_run.yaml`](imaging_preprocessing/configs/example_run.yaml),
edit it for your experiment, and pass it with `--config`:

```bash
cd imaging_preprocessing

# 1. Suite2p ROI extraction
python run_suite2p.py --config configs/my_experiment.yaml

# 2. Neuropil correction + ΔF/F
python compute_dff.py --config configs/my_experiment.yaml
```

A single config drives both stages. See the comments in the example file for
every available option (which mice/sessions, Suite2p `ops` overrides, ΔF/F
parameters, folder layout, and storage roots).

Running a script **without** `--config` reproduces the previous hardcoded
behaviour, so existing usage is unchanged.

### Shared settings

`imaging_preprocessing/config.py` holds the lab-wide defaults shared by both
scripts — the experimenter-initials map and the data/analysis folder roots
(lab server vs HAAS mounts). Add a new lab member there once, or override per
run via the config's `experimenter_map:` / `paths:` blocks.
