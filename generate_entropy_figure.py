"""
Generate the Approximate Entropy topographic figure comparing
psilocybin vs control conditions using corrected electrode locations.
"""
import json
import scipy.io
import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats.multitest import fdrcorrection
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── 1. Load corrected electrode locations ──────────────────────────────────
with open('corrected_electrode_locations.json', 'r') as f:
    electrode_locs = json.load(f)

# ── 2. Load and process approximate entropy data from .mat ─────────────────
mat_data = scipy.io.loadmat('approximate entropy.mat')
approx = mat_data['approx']

df = pd.DataFrame(approx)

def remove_brackets(x):
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.item()
    return x

df = df.map(remove_brackets)
df.columns = df.iloc[0].str.replace('.set', '', regex=False)
df = df.iloc[1:].reset_index(drop=True)

# ── 3. Extract electrode labels ────────────────────────────────────────────
electrodes = df['Approximate Entropy'].tolist()

# ── 4. Build case-insensitive mapping from .mat electrode names to JSON names
mat_to_json = {}
json_lower_map = {k.lower(): k for k in electrode_locs}
for e in electrodes:
    key = json_lower_map.get(e.lower())
    if key:
        mat_to_json[e] = key

# Ordered list of channel names as they appear in the JSON (matched to .mat order)
ch_names_json = [mat_to_json[e] for e in electrodes if e in mat_to_json]
ch_names_mat  = [e for e in electrodes if e in mat_to_json]

print(f"Matched {len(ch_names_json)}/{len(electrodes)} electrodes")

# ── 5. Create MNE montage from corrected locations ────────────────────────
positions_3d = {name: np.array([electrode_locs[name]['x'],
                                 electrode_locs[name]['y'],
                                 electrode_locs[name]['z']])
                for name in ch_names_json}

montage = mne.channels.make_dig_montage(ch_pos=positions_3d, coord_frame='head')
info = mne.create_info(ch_names=ch_names_json, sfreq=256, ch_types='eeg')
info.set_montage(montage)

# ── 6. Split data into ctrl / psi and set electrode index ─────────────────
df_all_ctrl = df.iloc[:, 1:233]
df_all_psi  = df.iloc[:, 233:]

# ── 7. Reorganize for paired analysis ─────────────────────────────────────
def reorganize_data(df_full):
    ctrl_cols = [c for c in df_full.columns if c.startswith('ctrl_sub')]
    psi_cols  = [c for c in df_full.columns if c.startswith('psi_sub')]

    def get_subject_activity(col):
        parts = col.split('_')
        return (parts[1], parts[2])

    ctrl_pairs = {get_subject_activity(c) for c in ctrl_cols}
    psi_pairs  = {get_subject_activity(c) for c in psi_cols}
    common     = ctrl_pairs & psi_pairs

    matched_ctrl = sorted([c for c in ctrl_cols if get_subject_activity(c) in common])
    matched_psi  = sorted([c for c in psi_cols  if get_subject_activity(c) in common])

    return df_full[matched_ctrl].copy(), df_full[matched_psi].copy()

df_ctrl, df_psi = reorganize_data(df)
df_ctrl.index = electrodes
df_psi.index  = electrodes
print(f"Paired data — ctrl: {df_ctrl.shape}, psi: {df_psi.shape}")

# ── 8. Paired t-tests with FDR ────────────────────────────────────────────
def perform_paired_comparison(ctrl, psi, elecs):
    pvals = []
    for e in elecs:
        _, p = scipy.stats.ttest_rel(
            ctrl.loc[e].astype(float),
            psi.loc[e].astype(float),
            nan_policy='omit')
        pvals.append(p)
    pvals = pd.Series(pvals, index=elecs)
    _, fdr = fdrcorrection(pvals)
    return pd.Series(fdr, index=elecs)

def get_activity_data(frame, activity):
    return frame.loc[:, [c for c in frame.columns if activity in c]]

result_matrix = pd.DataFrame(index=electrodes)
result_matrix['Electrodes'] = electrodes
result_matrix['Overall'] = perform_paired_comparison(df_ctrl, df_psi, electrodes)

for act in ['meditation', 'video', 'restingstate', 'music']:
    ca = get_activity_data(df_ctrl, act)
    pa = get_activity_data(df_psi, act)
    if not ca.empty and not pa.empty:
        result_matrix[act.capitalize()] = perform_paired_comparison(ca, pa, electrodes)

result_matrix.rename(columns={'Restingstate': 'Resting State'}, inplace=True)

pvalue_df = result_matrix.drop(columns='Electrodes').astype(float)
pvalue_df.index = electrodes

# ── 9. Helper functions for plotting ──────────────────────────────────────
def prepare_data(data, ch_names_mat_order):
    """Reorder data from .mat electrode order to JSON channel order."""
    idx = [electrodes.index(e) for e in ch_names_mat]
    return data[idx]

def calculate_difference(psi_df, ctrl_df):
    return psi_df.mean(axis=1) - ctrl_df.mean(axis=1)

def plot_topomap(data, info, ax, title, vmin, vmax, cmap):
    im, _ = mne.viz.plot_topomap(data, info, axes=ax, show=False,
                                  cmap=cmap, sensors=True, contours=6,
                                  vlim=(vmin, vmax))
    ax.set_title(title, fontsize=20)
    return im

# ── 10. Build figure ──────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman'] + plt.rcParams['font.serif']

apen_cmap = LinearSegmentedColormap.from_list(
    "custom_apen", ['#4B0082', '#8B008B', '#C0C0C0', '#FFA500', '#FFFF00'], N=100)
pvalue_cmap = LinearSegmentedColormap.from_list(
    "custom_pvalue", ['#FF0000', '#FF8000', '#FFFF00', '#FFFFFF', '#E6E6E6'], N=100)

fig = plt.figure(figsize=(26, 32))
gs  = fig.add_gridspec(5, 7,
                       width_ratios=[0.5, 10, 0.5, 1, 10, 0.5, 1],
                       height_ratios=[1, 1, 1, 1, 1])

activities = ['Overall', 'Meditation', 'Video', 'Resting State', 'Music']

# Also set index on the full ctrl/psi splits for difference calculation
df_all_ctrl.index = electrodes
df_all_psi.index  = electrodes

apen_vmin, apen_vmax     = -0.125, 0.125
pvalue_vmin, pvalue_vmax = pvalue_df.min().min(), 0.1

fig.text(0.31, 0.92, 'Approximate Entropy', fontsize=24, ha='center')
fig.text(0.71, 0.92, 'Statistical Significance', fontsize=24, ha='center')

for i, activity in enumerate(activities):
    fig.text(0.18, 0.89 - i * 0.159, chr(65 + i),
             fontsize=25, ha='center', va='center', weight='bold')

    # ── ApEn difference data ──
    if activity == 'Overall':
        diff = calculate_difference(df_psi, df_ctrl)
    else:
        act_key = activity.lower().replace(" ", "")
        psi_cols  = [c for c in df_psi.columns  if act_key in c.lower()]
        ctrl_cols = [c for c in df_ctrl.columns if act_key in c.lower()]
        diff = calculate_difference(df_psi[psi_cols], df_ctrl[ctrl_cols])

    apen_data = prepare_data(diff.to_numpy().reshape(-1), ch_names_mat)

    # ── P-value data ──
    pval_data = prepare_data(pvalue_df[activity].values, ch_names_mat)

    # ── Plot ApEn difference ──
    ax_apen  = fig.add_subplot(gs[i, 1])
    im_apen  = plot_topomap(apen_data, info, ax_apen, activity,
                             apen_vmin, apen_vmax, apen_cmap)
    cbar_ax  = fig.add_subplot(gs[i, 2])
    cbar     = fig.colorbar(im_apen, cax=cbar_ax, aspect=10)
    cbar.set_label('Difference in\n Psilocybin from Control', fontsize=18)

    # ── Plot p-values ──
    ax_pval  = fig.add_subplot(gs[i, 4])
    im_pval  = plot_topomap(pval_data, info, ax_pval, activity,
                             pvalue_vmin, pvalue_vmax, pvalue_cmap)
    cbar_ax2 = fig.add_subplot(gs[i, 5])
    cbar2    = fig.colorbar(im_pval, cax=cbar_ax2, aspect=10)
    cbar2.set_label('P-value', fontsize=18)
    cbar2.ax.axhline(y=0.05, color='k', linestyle='--', linewidth=2)

plt.savefig('eeg_topomap_plot.png', dpi=300, bbox_inches='tight')
print("Figure saved to eeg_topomap_plot.png")
