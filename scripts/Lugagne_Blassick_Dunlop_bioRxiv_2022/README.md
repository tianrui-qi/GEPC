The scripts in this folder were used to generate the various figures in our original Deep MPC paper.
The processed data necessary to reproduce the figures can be found on zenodo (see paper). We note that for some figure panels
were microscopy images are shown, access to the raw microscopy images is necessary. We can not provide these raw data (>1TB) 
as part of the zenodo archive.

Specifically, the following figures can be plotted from these scripts:
* `figure1.py`: Fig. 1 C, D
* `figure2.py`: Fig. 2 A, B, D-G, S3, S4
* `figure3.py`: Fig. 3 A-G, S9-13, Movies S1, S2
* `figure4.py`: Fig. 4 A-F, S14, Movie S3
* `figure5.py`: Fig. 5 B-E, S15-S18, Movie S4
* `revisions_analysis.py`: Fig. S1, S2, S5, S6
* `ode_mpc.py`: Fig. 1 H, S7, S8, Tables S2-S4
* `linear_predictions.py`: Table S1

The only thing that should have to be changed in these scripts are the paths at the top of each file.

Additionally, we include as reference `revisions_training.py` which was used to train the models analyzed in 
`revisions_analysis.py` on Boston University's Shared Computing Cluster.

We also provide `environment.txt` as a reference for the library versions that were installed in our conda environment when we
generated these figures. We had to use DeLTA for some figure panels (commit 8ceb015).
