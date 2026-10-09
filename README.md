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
python compute_dff_updated.py --config configs/my_experiment.yaml
```

A single config drives both stages. See the comments in the example file for
every available option (which mice/sessions, Suite2p `ops` overrides, ΔF/F
parameters, folder layout, and storage roots).

Running a script **without** `--config` reproduces the previous hardcoded
behaviour, so existing usage is unchanged.

### ΔF/F: `compute_dff_updated.py` vs `compute_dff.py`

`compute_dff_updated.py` is the ΔF/F script to use. `compute_dff.py` is the
old method, kept for comparison. Both run from the same config but read
different blocks, because their parameters (notably `window`) do not mean the
same thing.

| | `compute_dff.py` (old) | `compute_dff_updated.py` |
|---|---|---|
| Config block | `dff:` | `dff_updated:` |
| Neuropil correction | `F_cor = F - 0.7·Fneu`, clipped at 0 | `F_cor = F - 0.7·Fneu`, not clipped |
| Baseline F0 | 1 Hz FIR lowpass → rolling min/max filter (`window` = 30 s **per side**) → gaussian smoothing (5 s) | gaussian smoothing (1 s) → rolling **8th percentile** (`window` = 60 s **total**) |
| ΔF/F | `(F_cor − F0_cor) / F0_raw`: divided by the baseline of the **raw** trace | `(F_cor − F0) / F0`: divided by the baseline of the **neuropil-corrected** trace (F0 floored at 1) |
| Motion artifacts | ignored | suite2p `badframes` merged (< 2 s apart) and padded (2 s); excluded from F0 and set to NaN in `dff.npy` / `F_cor.npy` |
| Merged ROIs | set to non-cell in memory only | set to non-cell and **written to `iscell.npy`** (original kept as `iscell_original.npy`) |
| Saved traces | `F_raw`, `F_neu`, `F0_raw`, `F0_cor`, `dff` | `F_raw`, `F_neu`, `F_cor`, `F_cor_bad`, `F0`, `dff`, `dff_bad`, `artifact_frames` |
| QC | none | `noise_metrics.csv` + one `.npy` per metric (`noise_abs`, `noise_rel`, `snr`, `neuropil_ratio`, `neuropil_f_rho`, `contamination`), `dff_check.png`, `noise_distributions.png` |

Things to keep in mind when switching:

- **ΔF/F values are not comparable between the two scripts.** Dividing by the
  corrected baseline gives larger values, especially for cells with strong
  neuropil. Cells whose corrected baseline is close to 0 get very large values;
  the script logs them (F0 < 10% of raw F).
- `dff.npy` now contains NaN on artifact frames: use NaN-aware functions
  (`np.nanmean`, …) or load `dff_bad.npy` for the unmasked trace.
- `F0_raw.npy` and `F0_cor.npy` are no longer written.
- Both scripts skip sessions that already have a `dff.npy` unless `overwrite: true`.

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
