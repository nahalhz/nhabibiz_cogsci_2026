# nhabibiz_cogsci_2026

# Modeling Parkinsonian Freezing and DBS Effects in a Basal Ganglia Network

This repository contains the scripts and code used in the experiments reported in Habibizadeh, N. & Bartlett, M. (2026). *Modeling Parkinsonian Freezing and Deep Brain Stimulation Effects in a Basal Ganglia Network*. Preprint available on bioRxiv.

Note: this repository builds on the Dynamic Neural Field basal ganglia (DNF-BG) model from Bartlett, M., Furlong, M., Stewart, T. C., & Orchard, J. (2024). [Using Vector Symbolic Architectures for Distributed Action Representations in a Spiking Model of the Basal Ganglia](https://escholarship.org/content/qt6067f4sm/qt6067f4sm_noSplash_f1a0da7290d2c17947b90d550b3bc6c1.pdf). In Proceedings of the Annual Meeting of the Cognitive Science Society (Vol. 46), and relies on the dependency `sspspace` available at https://github.com/ctn-waterloo/sspspace.

## Replicate Experiments


**Study 1: Parkinsonian Freezing Model**
1. Run the grid search over thalamic threshold ($\theta$) and DNF scaling factor ($\alpha$) across healthy and dopamine-depleted conditions and get the results as a csv  — `>> python study1.py`
2. Generate results tables/csv files saved as — ` study1_results.csv`

**Study 2: Deep Brain Stimulation Exploratory Study**
3. Run the DBS grid search over STN ($w_t$) and GPi ($w_p$) scaling, repeated across $n=20$ experiments per configuration — `>> python study2.py`
4. Generate results tables/csv files saved as — ` study2_results.csv`

You will be prompted to provide the paths to the data and to the figures. Data will be saved as npz files and then as csv files. The csv files will be accessed for analysis and plotting.

All heatmaps and figures reported in the paper (selection-ratio heatmap, top-configuration bar/box plots, DBS success heatmap, and DBS comparison bar plot) are generated in the `results_produced.ipynb` notebook, which loads the csv files produced above.

## Networks

The networks include the thalamic gating mechanism implemented using `nengo.networks.Thalamus`, copied/adapted from https://github.com/nengo/nengo/blob/main/nengo/networks/actionselection.py. It is provided here under a GPLv2 license.

We coupled this thalamic network to the DNF-BG model (striatal D1/D2, STN, GPe, GPi populations) to enable explicit measurement of action gating success, and introduced a global DNF scaling parameter ($\alpha$) and DBS afferent-scaling parameters ($w_t$, $w_p$) to model dopamine depletion and stimulation effects, respectively.

---

To cite this project please use:
```bibtex
@article{habibizadeh2026modeling,
  title={Modeling Parkinsonian Freezing and Deep Brain Stimulation Effects in a Basal Ganglia Network},
  author={Habibizadeh, Nahal and Bartlett, Madeleine},
  journal={bioRxiv},
  year={2026}
}
```
