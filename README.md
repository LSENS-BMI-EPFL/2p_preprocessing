# 2p_data_preprocessing

Preprocessing pipeline for 2-photon calcium imaging data (Suite2p extraction +
ΔF/F computation).

## Configuration

Run parameters live in a **YAML config file** instead of being hardcoded in the
scripts. Copy [`imaging_preprocessing/configs/my_experiment.yaml`](imaging_preprocessing/configs/my_experiment.yaml),
edit it for your experiment, and pass it with `--config`:

```bash
cd imaging_preprocessing

# 1. Suite2p ROI extraction
python run_suite2p.py --config configs/my_experiment.yaml

# 2. Neuropil correction + ΔF/F
python compute_dff_updated.py --config configs/my_experiment.yaml

# 3. Spike inference (CASCADE) + quantification, in the Cascade env
conda activate Cascade
python run_cascade_inference.py --config configs/my_experiment.yaml
```

A single config drives all stages. See the comments in the example file for
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
| Motion artifacts | ignored | frames where the registration correlation `corrXY` drops below 0.7 × its 20 s rolling median (merged < 0.2 s, padded 0.1 s: the disturbance fades 3-4 frames after the flagged frames whatever the period length), plus sudden whole-FOV brightness steps > 1.5x (e.g. immersion water refilled, PMT settling): F0 lags them for half its window, so 30 s on the bright side is masked; excluded from F0 and set to NaN in `dff_masked.npy` / `F_cor_masked.npy` (same frames for every cell). suite2p `badframes` are not used: they all fall inside these periods |
| Merged ROIs | set to non-cell in memory only | set to non-cell and **written to `iscell.npy`** (original kept as `iscell_original.npy`) |
| Saved traces | `F_raw`, `F_neu`, `F0_raw`, `F0_cor`, `dff` | `F_raw`, `F_neu`, `F_cor`, `F_cor_masked`, `F0`, `dff`, `dff_masked`, `artifact_frames` |
| QC | none | `noise_metrics.csv` + one `.npy` per metric (`noise_abs`, `noise_rel`, `snr`, `neuropil_ratio`, `neuropil_ratio_max`, `neuropil_f_rho`, `contamination`, `motion_r`), `dff_check.png`, `noise_distributions.png` |

Things to keep in mind when switching:

- **ΔF/F values are not comparable between the two scripts.** Dividing by the
  corrected baseline gives larger values, especially for cells with strong
  neuropil. Cells whose corrected baseline is close to 0 get very large values;
  the script logs them (F0 < 10% of raw F).
- `dff.npy` keeps artifact frames. `dff_masked.npy` has them set to NaN: use
  NaN-aware functions (`np.nanmean`, …) on it.
- **suite2p `badframes` miss most motion artifacts.** Its `th_badframes` is
  normalized by the session's mean shift, so frequent jumps never stand out.
  The `corrXY` criterion (`motion_*` options) catches them, including every
  suite2p bad frame. Share of frames masked with 0.1 s padding: ~5% in
  RD130_20260929, ~18% in RD134_20260929 (more task-evoked movement); with
  0.25 s padding the 38-session survey gave ~8-9% per session in RD130/RD132
  and ~22% in RD134. Cells already deviate more than
  usual at corrXY 0.8-0.9 x local median, so 0.7 is not a strict threshold.
- **Cell-level QC is decided at analysis, not here.** Every `iscell == 1` cell is
  kept. Useful cuts on the saved metrics:
  - `neuropil_ratio_max` > ~0.8: in at least one 60 s window, neuropil
    subtraction removed most of the baseline, F0 drops to its floor and dF/F
    explodes (single cells can produce most of a session's "10-20 spike"
    events, at 10,000% dF/F). Use it rather than the session `neuropil_ratio`:
    a cell whose neuropil ring brightens for a few minutes (e.g. an active
    undetected neighbour) collapses there while its session median looks fine.
    Such cells also get an inflated SNR, so a low-SNR cut does not remove them.
  - |`motion_r`| large: dF/F follows registration quality frame by frame, i.e. a
    motion artifact. Real population activity also lowers `corrXY`, so
    moderate values are ambiguous.
  - `contamination` = `neuropil_ratio` × `neuropil_f_rho` is negative when F and
    F_neu are anticorrelated (often motion), so check `neuropil_ratio` as well.
- **Movie snippets** (`movie_snippets: true`): short clips of the registered
  movie (`reg_tif`) are saved to `suite2p/plane0/calcium_movie_snippets/`.
  `<session>_movement_example_<i>.tiff` (`snippet_n_movement`, default 8) covers a
  motion period from `snippet_t_before` s before to `snippet_t_after` s after,
  at most `snippet_max_frames` (300) frames. Periods are picked evenly from the
  deepest to the mildest `corrXY` drop. `<session>_stable_example_<i>.tiff`
  (`snippet_n_stable`, default 2) is a 300-frame stretch with no artifact
  frame. `snippets.csv` gives the frame range of each clip. Frames are mapped
  to the tif files with `reg_tif/movie_specs.yaml` if present, else with the
  registration `batch_size` in `ops.npy`. Snippets are redone on every run.
- `F0_raw.npy` and `F0_cor.npy` are no longer written.
- Both scripts skip sessions that already have a `dff.npy` unless `overwrite: true`.

### Spike inference: `run_cascade_inference.py`

Runs [CASCADE](https://github.com/HelmchenLabSoftware/Cascade) on `dff.npy`
(NaN-free, artifact frames kept), then plots quantification figures. Settings
are in the config's `cascade:` block.

**It needs its own environment.** CASCADE installed from git needs python 3.7
and tensorflow 2.3 (numpy 1.18), which can't be installed alongside the latest
suite2p. Steps 1–2 run in the suite2p env and step 3 runs in the `Cascade` env:

```bash
git clone https://github.com/HelmchenLabSoftware/Cascade
# conda only provides python; the packages come from pip (a conda solve of the full list is very slow).
conda create -n Cascade python=3.7
conda activate Cascade
pip install tensorflow==2.3.0 matplotlib seaborn ruamel.yaml pyyaml tqdm
```

`tensorflow==2.3.0` pulls in the numpy/scipy/h5py versions it needs
(numpy 1.18, scipy 1.4.1, h5py 2.10), so don't pin or upgrade them separately.

- `cascade2p` is not pip-installed. The script imports it from
  `cascade.cascade_repo` and runs from that folder, because CASCADE looks for
  `Pretrained_models/` and `Ground_truth/` relative to it.
- `cascade_repo` must be reachable from the machine running the script. To run
  on HAAS, clone CASCADE on the analysis share and use its `/mnt/lsens-analysis/...`
  path. The `Cascade` env itself still has to be created on HAAS. The model is
  downloaded into the clone on the first run, so that run needs internet access.
- The Cascade env can't read `ops.npy` (it was pickled with numpy 2), so the
  frame rate comes from `cascade.frame_rate`.
- `config.py` only needs `pyyaml`, so both environments share it.

Outputs go to `<suite2p folder>/<model_name>/`:

| File | Content |
|---|---|
| `spike_rate_prediction.npy` | CASCADE output, expected spikes **per frame** (× frame rate = spk/s), NaN at the edges |
| `discrete_spikes.npy` | object array, spike frame indices per cell, as returned by CASCADE: **1 frame early**, add 1 (`allow_pickle=True`) |
| `spike_counts.npy` | discrete spikes binned per frame, `(n_cells, n_frames)` int, 1-frame offset corrected |
| `noise_level_distribution.png`, `noise_matched_ground_truth.png` | CASCADE QC for model choice |
| `n_spikes_events_hist.png` | events per spike count (spiking frames < `gap_s` apart are merged) |
| `dff_single_by_n_spikes.png` | single ΔF/F responses to isolated frames, `n_single_examples` per spike count, with the mean overlaid |
| `dff_avg_by_n_spikes.png` | mean ± sem ΔF/F, all events vs isolated frames |
| `dff_linearity_isolated.png` | peak ΔF/F vs spike count, fit through the origin |
| `example_cells_f_dff_rate_spikes.png` | F raw, ΔF/F, rate and spike counts for example cells (`snr.npy` ≥ `example_snr_min`) |

The quantification doesn't need NWB. Time is frame index / frame rate, with
0 at imaging start. Saved predictions are reused unless `overwrite: true`.
The figures are redone on every run.

**Artifact frames and CASCADE.** CASCADE runs on the full `dff.npy`: it can't
take NaN (`infer_discrete_spikes` would join the pieces around a gap and shift
every later spike). Two things keep artifacts out:

- Each cell's noise level, which selects the model ensemble and so calibrates
  the spike rate, is computed on `dff_masked.npy` (artifact frames ignored).
- The quantification figures leave out cells with `neuropil_ratio_max` above
  `dff_updated: neuropil_ratio_threshold` (0.8) or `contamination` above
  `dff_updated: contamination_threshold` (0.8): a collapsed baseline makes
  dF/F explode, and a single such cell can dominate the large-event averages.
  Their spikes are still inferred and saved.
- The quantification leaves out spikes on artifact frames, plus
  `spike_exclusion_pad` seconds around them (default 0). Each prediction looks
  at a 64-frame window (±1.07 s at 30 Hz), so spikes next to an artifact can be
  affected too, but excluding that full window would drop a large share of the
  spikes in sessions with frequent movement.

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
