# Tools

This folder contains (one of the) tools that were used to produce the dataset.

These are the steps that were taken:
1. <unpublished> create the raw dataset `data_raw.ldjson`.
2. Run `raw2masked.py`. This tool resets timestamps, replace domains and feature names with generic IDs (and store the mapping in JSON dictionaries), and create a balanced data-set as a stratified sample over the "target_verdict".
```bash
    ./raw2masked.py -vv -i ../../data/data_raw.ldjson -p ../../data/pub_
```
3. Split, deduplicate, recombine. This will make sure very common patterns are repeated (e.g. once per split) but not more.
```bash
    split -l 60000 pub_1st_balanced_flat.ldjson
```
4. Create a DataFrame in that can be quickly loaded in python.
```python
import pandas as pd
from glob import glob
splits = glob('./xa[a-z]')
fullDF_ext = pd.DataFrame()

# Iterate over splits and ded    df = pd.read_json(fn, lines=True)
    df = df.fillna(0)
    split_rows = df.shape[0]
    # Consider adding more fields such as 'suffix'
    df_ext = df.drop_duplicates(subset=df.filter(regex='target_verdict|feature\.|align|pass',
                                                 axis=1).columns,
                                ignore_index=True)
    split_unique_rows = df_ext.shape[0]
    # Recombine with data from previous splits.
    fullDF_ext = pd.concat([fullDF_ext, df_ext], ignore_index=True)
    print(f'{fn}: {split_rows} rows, {split_unique_rows} '
          f'unique, fullDF_ext now {fullDF_ext.shape}')

# Replace nan (might have been a missing column in a given split) with 0
#   and delete columns with only 0 values.
fullDF_ext = fullDF_ext.fillna(0)
fullDF_ext = fullDF_ext.loc[:, (fullDF_ext != 0).any(axis=0)]
print(f'deleted "0"-columns, fullDF_ext now {fullDF_ext.shape}')


# Write deduplicated data to file.
# Sort by queryTS.
fullDF_ext.sort_values(['queryTS'], ignore_index=True, inplace=True)
fullDF_ext.to_pickle('uniqueish_ext.pkl')
unique_rows = fullDF_ext.drop_duplicates(subset=fullDF_ext.filter(regex='target_verdict|feature\.|align|enabled', axis=1).columns).shape[0]
print(f'Wrote {fullDF_ext.shape[0]} records, {unique_rows} unique in verdict and features.')
```

