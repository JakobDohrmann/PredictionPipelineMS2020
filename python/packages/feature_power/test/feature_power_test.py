import pandas as pd
from lightgbm import LGBMRegressor
from time import time

# Load data
fullDF = pd.read_pickle('../../../data/deduped_dataframe.pkl')
X = fullDF.filter(regex='feature\.|(spf|dkim|dmarc)\.(align|pass)', axis=1).astype('int')
yS = fullDF['target_score'].astype('float')

RAND_STATE = 27
lgbm_base_params = {'random_state': RAND_STATE,
                    'learning_rate': 0.3,
                    'num_leaves': 31,
                    'n_estimators': 750,
                    'reg_lambda': 1}

# (Re-) train a model
rgbm = LGBMRegressor()
rgbm.set_params(**lgbm_base_params)
rgbm.fit(X, yS)
jgbm = rgbm._Booster.dump_model(num_iteration=-1)
num_features = len(jgbm['feature_names'])

# Warming up LRU cache
for i in range(1000):
    _ = _lru_fac(i)

for inc in [True, False]:
    for cs in range(1, int(num_features / 2)):
        _ = _coalition_quotient_numerator(cs, num_features, inc)

# Feature Power
first_tree_default_prediction = rgbm.predict(X.iloc[0:1, :], pred_contrib=True)[-1][-1]
discrimination_threshold = .5  # aggressive

t0 = time()
_ = regrGBM_FP_agg(jgbm,  # json representation of the model
                   'pathPow',  # pathPow, cumNodePow, strictNodePow
                   first_tree_default_prediction,  # f_0
                   discrimination_threshold)  # \rho
print(f'Done with FP_pp, took {int(time() - t0)} seconds')

t0 = time()
_ = regrGBM_FP_agg(jgbm,
                   'cumNodePow',
                   first_tree_default_prediction,
                   discrimination_threshold)
print(f'Done with FP_cn, took {int(time() - t0)} seconds')

t0 = time()
_ = regrGBM_FP_agg(jgbm,
                   'strictNodePow',
                   first_tree_default_prediction,
                   discrimination_threshold)
print(f'Done with FP_sn, took {int(time() - t0)} seconds')
