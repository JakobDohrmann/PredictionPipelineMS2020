# Data
This folder contains the dataset use in the thesis (loaded in the jupyter notebook).
Raw data is too large for github (~300MB) and will be hosted in a dedicated location.
Links to the download location will be made available here.

Files are compressed using GNU zip. Decompress using `gunzip` or any decent archiving program.

* `pub_balanced_raw.ldjson`: Stratified, 1.2M data points, raw data, line-delimited nested json, [download raw data from BIDAL](https://bidal.sfsu.edu/~kazokada/research/pub_balanced_raw.ldjson.gz) 
* `pub_balanced_flat.ldjson`: Stratified, 1.2M data points, one-hot encoded (252 binary features), line-delimited json, [download flat data from BIDAL](https://bidal.sfsu.edu/~kazokada/research/pub_balanced_flat.ldjson.gz)
* `deduped_dataframe.pkl`: This was used in experiments. Reduced duplication, 130K data points, pandas DataFrame
* `noise_results.pkl`: Results for experiment 5.6.3.1 Robustness Against Selecting Highly Correlated Features, since they are too large and not of general interest when printed out directly in the notebook.

