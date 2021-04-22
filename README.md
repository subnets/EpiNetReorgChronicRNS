# EpiNetReorgChronicRNS
Code and analysis notebooks to accompany Khambhati, Shafi, Rao and Chang "Long-term brain network reorganization predicts responsive neurostimulation outcomes for focal epilepsy".


## Installation
We recommend creating a new virtual environment using `Anaconda` or `virtualenv` before proceeding with this installation.
```
git clone https://github.com/subnets/EpiNetReorgChronicRNS.git
cd EpiNetReorgChronicRNS
pip install -r requirements.txt
```
Edit the `notebooks/paths.json` file to point to directories containing raw data and output data.

## Data Processing
Run the `notebooks/batch_preproc.sh` script to execute the data processing pipeline. In practice, this pipeline will take several hours to days to run depending on the number of data clips in the database. To manually run this pipeline, simply run each `python <execfile.py>` command separately. When run in full, the following steps will be executed:

1. Detection of stimulation periods.
2. Line Length feature extraction.
3. Alternate spike extraction methods.
4. Wavelet convolution and functional connectivity estimation.
5. Non-Negative Tensor Factorization of functional connectivity data. 

## Analysis Notebooks
Notebooks are run using Jupyter Notebook, and the main analyses of each notebook is summarized below:
1. Clinical characteristics of the cohort (Fig. 1, Fig. S1, Fig. S5) [`e000-ANK-Clinical_Covariates.ipynb`]
2. Plotting and Visualization of iEEG recorded from the device (Fig. 1, Fig. S2) [`e000-ANK-Stim_Detect.ipynb`]
3. Detection of interictal epileptiform activity (Fig. S3) [`e001-ANK-Line_Length.ipynb`]
4. Comparison of interictal epileptiform activity and interictal functional connectivity (Fig. S4) [`e002-ANK-Phase_Locking_Value-LL_Correlation.ipynb`]
5. Reorganization of neocortical functional connectivity (Fig. 2, Fig. S6) [`e002-ANK-Phase_Locking_Value-NEO.ipynb`]
6. Reorganization of mesial temporal functional connectivity (Fig. 2, Fig. S7) [`e002-ANK-Phase_Locking_Value-MTL.ipynb`]
7. Reorganization of functional connectivity outside seizure focus (Fig. S8) [`e002-ANK-Phase_Locking_Value-Bridging.ipynb`]
8. Dynamical prediction horizon of outcome (Fig. 3, Fig. S9) [`e003-ANK-Prediction_Horizon.ipynb`]
9. Cumulative effect of stimulation on network reorganization (Fig. 4, Fig. 5, Fig. S10, Fig. S11, Fig. S12) [`e004-ANK-PLV_NTF.ipynb`]
10. Change in functional connectivity precedes reduction in electrographic seizures (Fig. S13) [`e004-ANK-PLV_NTF.ipynb`]
