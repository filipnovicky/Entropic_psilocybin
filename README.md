# Psilocybin EEG: microstates and approximate entropy

Supporting code and derived data for

> Novický, F., Stoliker, D., Razi, A. & Zeldenrust, F.
> *EEG microstates change more rapidly and approximate entropy
>  increases under acute psilocybin*

The analyses run on the PsiConnect dataset: 63 participants recorded in a
baseline session and again ~200 min after 19 mg of psilocybin, in four
conditions (video watching, eyes-closed rest, guided meditation, music
listening). 429 recordings in total.

## The two notebooks

| Notebook | What it does |
|---|---|
| [`microstates.ipynb`](microstates.ipynb) | Quality control, statistics and Figure 1, from the MICROSTATELAB exports |
| [`approximate_entropy.ipynb`](approximate_entropy.ipynb) | ApEn implementation, statistics and Figure 2 |

Both are self-contained — all code lives in the notebooks themselves — and both
read only from `data/` and `results/`, so they run end to end without the raw
recordings.

## Data availability

The raw EEG recordings are not part of this repository. The PsiConnect dataset
is openly available at
<https://openneuro.org/datasets/ds006110/versions/1.2.0>. The preprocessed
("clean") version used for the analyses here can be obtained by contacting the
corresponding author.

The derived data needed to reproduce every number and figure below *are*
included, all of it under `data/`: the MICROSTATELAB exports, the montage, and
approximate entropy per electrode and recording. Both notebooks therefore run
end to end from `data/` alone, without the recordings.

## Method notes

Details are in the paper; these are the choices a reader of the code may want
confirmed.

- **Approximate entropy** follows Pincus (1991) with `m = 2`, `r = 0.2`,
  non-overlapping 5 s windows, averaged over windows within each electrode.
  Each window is normalised to zero mean and unit standard deviation so the
  tolerance applies on the scale of the data being compared; because ApEn is
  scale invariant this is identical to using `r · σ` on the unnormalised
  window. Template distances use the Euclidean norm; setting `METRIC =
  "chebyshev"` selects the maximum norm of Pincus (1991) instead.
- **Sampling rate.** Recordings are 500 Hz and are decimated to 125 Hz before
  ApEn; microstate analysis uses the original rate.
- **Missing electrodes are interpolated.** 396 of the 429 recordings are
  missing at least one electrode, so electrodes rejected in preprocessing are
  reconstructed from their neighbours by spherical spline; every recording then
  contributes all 64 and every electrode the same number of pairs. The notebook
  repeats the analysis with them left missing instead
  (`data/apen_complete_case.csv`) and the two agree.
- **Microstate quality control.** An individual topography sharing less than
  60% variance with its reference is dropped for that participant, along with
  the aggregate metrics that sum over all four microstates.

## Recomputing approximate entropy

`data/apen.csv` (64 electrodes × 429 recordings) ships with the repository, so
the notebook runs without the raw recordings. To recompute from them, obtain
the recordings (see Data availability), point `PSICONNECT_ROOT` at a directory
laid out as `{Control,Psilocybin}/{Meditation,Music,Resting state,Video}/*.set`,
and set `RECOMPUTE = True` in the configuration cell of
`approximate_entropy.ipynb`. That takes roughly 5 minutes at `N_JOBS = 8`.

## Layout

```
microstates.ipynb          quality control, statistics, Figure 1
approximate_entropy.ipynb  ApEn implementation, statistics, Figure 2
data/                      everything the notebooks read
results/                   statistics tables (generated)
figures/                   figures (generated)
```

### `data/`

| File | Contents |
|---|---|
| `df_microstates.csv` | 30 microstate metrics per recording (MICROSTATELAB) |
| `shared_variance.csv` | Shared variance of each individual topography with the reference |
| `var_and_time.csv` | Total time and total explained variance per recording |
| `corrected_electrode_locations.json` | 3-D positions for the 64 electrodes |
| `eeg_template.fif` | Montage template |
| `apen.csv` | Approximate entropy, 64 electrodes × 429 recordings |
| `apen_complete_case.csv` | The same without interpolation, for the sensitivity check |

`df_microstates.csv`, `shared_variance.csv` and `var_and_time.csv` are direct
exports of the MICROSTATELAB plugin for EEGLAB and are not recomputed here.

## Requirements

Python ≥ 3.11 with `mne`, `numpy`, `scipy`, `pandas`, `statsmodels`, `seaborn`,
`matplotlib`, `joblib`.
