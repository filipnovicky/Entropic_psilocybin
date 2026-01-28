"""
Generate the Approximate Entropy topographic figure comparing
psilocybin vs control conditions using corrected electrode locations.

Outputs both PNG and PDF.
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
from matplotlib.gridspec import GridSpec

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

# ── 9. Helper functions ───────────────────────────────────────────────────
def prepare_data(data, ch_names_mat_order):
    idx = [electrodes.index(e) for e in ch_names_mat]
    return data[idx]

def calculate_difference(psi_df, ctrl_df):
    return psi_df.mean(axis=1) - ctrl_df.mean(axis=1)

# Also set index on the full ctrl/psi splits
df_all_ctrl.index = electrodes
df_all_psi.index  = electrodes

# ── 10. Build improved figure ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.linewidth': 0.8,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
})

apen_cmap = LinearSegmentedColormap.from_list(
    "custom_apen", ['#4B0082', '#8B008B', '#C0C0C0', '#FFA500', '#FFFF00'], N=256)
pvalue_cmap = LinearSegmentedColormap.from_list(
    "custom_pvalue", ['#FF0000', '#FF8000', '#FFFF00', '#FFFFFF', '#E6E6E6'], N=256)

activities = ['Overall', 'Meditation', 'Video', 'Resting State', 'Music']

apen_vmin, apen_vmax = -0.125, 0.125
pvalue_vmin = pvalue_df.min().min()
pvalue_vmax = 0.1

# ── Layout: 5 rows of topomaps, shared colorbars at bottom ───────────────
fig = plt.figure(figsize=(14, 20))

# Main grid: 6 rows (5 data + 1 colorbar), 2 columns
outer_gs = GridSpec(6, 2, figure=fig,
                    height_ratios=[1, 1, 1, 1, 1, 0.08],
                    width_ratios=[1, 1],
                    hspace=0.25, wspace=0.15,
                    left=0.08, right=0.92, top=0.93, bottom=0.06)

# Column headers
fig.text(0.30, 0.955, 'Approximate Entropy Difference',
         fontsize=16, ha='center', weight='bold')
fig.text(0.72, 0.955, 'Statistical Significance (FDR)',
         fontsize=16, ha='center', weight='bold')

for i, activity in enumerate(activities):
    # ── Compute data ──
    if activity == 'Overall':
        diff = calculate_difference(df_psi, df_ctrl)
    else:
        act_key = activity.lower().replace(" ", "")
        psi_cols  = [c for c in df_psi.columns  if act_key in c.lower()]
        ctrl_cols = [c for c in df_ctrl.columns if act_key in c.lower()]
        diff = calculate_difference(df_psi[psi_cols], df_ctrl[ctrl_cols])

    apen_data = prepare_data(diff.to_numpy().reshape(-1), ch_names_mat)
    pval_data = prepare_data(pvalue_df[activity].values, ch_names_mat)

    # ── Row label ──
    label = chr(65 + i)

    # ── ApEn topomap ──
    ax_apen = fig.add_subplot(outer_gs[i, 0])
    im_apen, cn_apen = mne.viz.plot_topomap(
        apen_data, info, axes=ax_apen, show=False,
        cmap=apen_cmap, sensors=True, contours=6,
        vlim=(apen_vmin, apen_vmax))
    cn_apen.set_alpha(0.25)
    cn_apen.set_linewidths(0.4)
    ax_apen.set_title(f'{label}.  {activity}', fontsize=15, pad=8,
                      loc='left', weight='bold')

    # ── P-value topomap ──
    ax_pval = fig.add_subplot(outer_gs[i, 1])
    im_pval, cn_pval = mne.viz.plot_topomap(
        pval_data, info, axes=ax_pval, show=False,
        cmap=pvalue_cmap, sensors=True, contours=6,
        vlim=(pvalue_vmin, pvalue_vmax))
    cn_pval.set_alpha(0.25)
    cn_pval.set_linewidths(0.4)
    ax_pval.set_title(f'{activity}', fontsize=15, pad=8,
                      loc='left', style='italic')

# ── Shared colorbars at the bottom ────────────────────────────────────────
cbar_ax_left = fig.add_subplot(outer_gs[5, 0])
cb1 = fig.colorbar(im_apen, cax=cbar_ax_left, orientation='horizontal')
cb1.set_label(r'$\Delta$ ApEn  (Psilocybin $-$ Control)', fontsize=13)
cb1.ax.tick_params(labelsize=11)

cbar_ax_right = fig.add_subplot(outer_gs[5, 1])
cb2 = fig.colorbar(im_pval, cax=cbar_ax_right, orientation='horizontal')
cb2.set_label('FDR-corrected p-value', fontsize=13)
cb2.ax.tick_params(labelsize=11)
# Draw significance threshold line on p-value colorbar
cb2.ax.axvline(x=0.05, color='k', linestyle='--', linewidth=1.5)
cb2.ax.text(0.05, 1.35, 'p = .05', transform=cb2.ax.transData,
            fontsize=10, ha='center', va='bottom')

# ── Save ──────────────────────────────────────────────────────────────────
for fmt in ['pdf', 'png']:
    outpath = f'eeg_topomap_plot.{fmt}'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved {outpath}")

plt.close(fig)
