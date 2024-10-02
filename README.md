This repository contains documents to reproduce analyses done for the paper 'silocybin-Induced Alterations in EEG Microstates and Signal Complexity: Evidence for the Entropic Brain Hypothesis by Filip Novicky et al.'. The following lines explain every file:

df_microstates.csv and shared_variance.csv are direct outputs from the MICROSTATELAB plugin in EEGLAB. These files are further processed in psilocybin_microstates.ipynb where statistical analyses and figures are produced.

For the approximate entropy, a new code was developed: ApproximateEtropy_EEGLAB.m. The output of this code is approximate entropy.mat, where this file is further processed in Approximate_Entropy.ipynb. Note that for visualization, you need to download eeg_template.fif.
