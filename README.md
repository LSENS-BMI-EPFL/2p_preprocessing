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

`imaging_preprocessing/config.py` holds the lab-wide defaults shared by all
scripts — the experimenter-initials map and the data/analysis folder roots
(lab server vs HAAS mounts). Add a new lab member there once, or override per
run via the config's `experimenter_map:` / `paths:` blocks.

## Projection-neuron GUI

The classification GUI is also config-driven (mice, paths, channel tags and
registration model live in YAML, not in the code):

```bash
cd imaging_preprocessing/projection_gui
python projection_gui.py                 # uses configs/example_gui.yaml
python projection_gui.py my_config.yaml  # uses your config
```

See [`projection_gui/configs/example_gui.yaml`](imaging_preprocessing/projection_gui/configs/example_gui.yaml).

## Session stitching / fixing

`imaging_preprocessing/session_stitching.py` is a library of reusable helpers
for fixing split or over-recorded sessions (reading `log_continuous.bin`,
detecting imaging/trial/camera events, truncating logs, stitching logs and
behaviour tables, counting/truncating/merging tiffs and avis). Import the
functions, or use the CLI for the common cases:

```bash
python session_stitching.py stitch-logs sess1/log_continuous.bin sess2/log_continuous.bin out/log_continuous.bin
python session_stitching.py truncate-log in.bin out.bin --at-last-trial
python session_stitching.py count-frames --log log.bin --tiff movie.tif
```
