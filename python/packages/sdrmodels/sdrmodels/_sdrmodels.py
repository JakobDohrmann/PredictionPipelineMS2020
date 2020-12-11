import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class SDRMeasures(object):
    """Custom measures and objectives for lightGBM or catboost:
https://catboost.ai/docs/concepts/pytho21n-usages-examples.html#user-defined-loss-function
https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMModel.html#lightgbm.LGBMModel
"""
    def __init__(self, lbd=10, objective='werr',
                 penalties={'-':{'-':3, '?':6, '+':11},
                            '?':{'-':4, '?':2, '+':7},
                            '+':{'-':5, '?':3, '+':1}}):
        self.lbd = lbd
        # Needs to be initialized so the catboost wrapper
        # (fixed/predefined name!) is pointed at the appropriate function.
        if objective == 'werr':
            self.objective = self.sdr_objective_werr
        else:
            raise ValueError('Objective not implemented.')
        self.penalties = penalties
        sum_pos = sum(penalties['+'].values())
        self.class_weights = {lbl:sum(preds.values())/sum_pos-1
                              for lbl,preds in penalties.items()}
        super().__init__()

    def sdr_objective_werr(self, y_true, y_pred, weights=None):
        """Objective for SDR GBM models. Emphasizing FP penalty.
Input:
    y_true: array-like of shape = [n_samples] - target values.
    y_pred: array-like of shape = [n_samples] - predicted values.

Returns:
    grad: array-like of shape = [n_samples] - value of the first order
          derivative (gradient) for each sample point.
    hess: array-like of shape = [n_samples] - value of the second order
          derivative (Hessian) for each sample point.
"""
        assert len(y_true) == len(y_pred), 'y_true,y_pred dimension mismatch'
        if weights is not None:
            assert len(weights) == len(y_pred), 'y,weight dimension mismatch'
        return self.calc_ders_range(y_pred, y_true, weights)

    def calc_ders_range(self, approxes, targets, weights=None):
        """catboost-compatible wrapper around sdr objective functions."""
        # approxes, targets, weights are indexed containers of floats
        # (containers which have only __len__ and __getitem__ defined).
        # weights parameter can be None.
        #
        # To understand what these parameters mean, assume that there is
        # a subset of your dataset that is currently being processed.
        # approxes contains current predictions for this subset,
        # targets contains target values you provided with the dataset.
        #
        # This function should return a list of pairs (der1, der2), where
        # der1 is the first derivative of the loss function with respect
        # to the predicted value, and der2 is the second derivative.

        # lightgbm and catboost have different parameter order and return
        # their results in slightly different formats...
        grad, hess = self.objective(targets, approxes, weights)
        return [pair for pair in zip(grad, hess)]

    def sdr_muliclass_objective_lightgbm(self, y_true, y_pred, weights=None):
        """
y_true: array-like of shape = [n_samples] - target values.
y_pred: array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)

Returns:
    grad: array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)

    The value of the first order derivative (gradient) for each sample point.
hessarray-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)

    The value of the second order derivative (Hessian) for each sample point.
"""
        # lightgbm and catboost have different parameter order...
        return self.calc_ders_range(y_pred, y_true, weights)

    def calc_ders_multi(self, approxes, target, weight):
        # approxes - indexed container of floats with predictions
        #            for each dimension of single object
        # target - contains a single expected value
        # weight - contains weight of the object
        #
        # This function should return a tuple (der1, der2), where
        # - der1 is a list-like object of first derivatives of the loss function with respect
        # to the predicted value for each dimension.
        # - der2 is a matrix of second derivatives.
        pass

    @staticmethod
    def sdr_rates_from_confusion(cf, mode='aggressive', lbd=1):
        """Return the TPR, FNR, FPR, TNR computed from a 3x3 confusion matrix.
Mode affects what is considered as a TP/FP:
'aggressive':   TPR=p([+,?],[+,?]), TNR=p(-,-),
'conservative': TPR=p(+,+),         TNR=p([-,?],[-,?]),
'threeclass':   TPR=p(+,+)+p(?,?),  TNR=p(-,-), (+,?)."""
        assert cf.shape == (3, 3), 'Expect a 3x3 confusion matrix'
        cf = cf.ravel()
        # normalize
        total_weight = sum(cf)
        cf = cf/total_weight
        if mode.lower().startswith('a'):
            tpr = sum(cf[[0,1,3,4]])
            fnr = sum(cf[[2,5]])
            fpr = sum(cf[[6,7]])
            tnr = cf[8]
        elif mode.lower().startswith('c'):
            tpr = cf[0]
            fnr = sum(cf[[1,2]])
            fpr = sum(cf[[3,6]])
            tnr = sum(cf[[4,5,7,8]])
        elif mode.lower().startswith('t'):
            tpr = sum(cf[[0,4]])
            fnr = sum(cf[[1,2,5]])
            fpr = sum(cf[[3,6,7]])
            tnr = cf[8]
        else: #weighted
            raise('TODO: not yet implemented')
        return (tpr, fnr, fpr, tnr)

    @staticmethod
    def sdr_measure_tcr(tpr, fnr, fpr, tnr, lbd=10):
        return (tpr+fnr) / (lbd*fpr + fnr)

    @staticmethod
    def sdr_measure_wacc(tpr, fnr, fpr, tnr, lbd=10):
        return (lbd*tnr + tpr) / (lbd*(tnr + fpr) + (tpr + fnr))

    @staticmethod
    def sdr_measure_werr(tpr, fnr, fpr, tnr, lbd=10):
        return (lbd*fpr + fnr) / (lbd*(tnr + fpr) + (tpr + fnr))


class _SDRLookupTable(BaseEstimator):
    """'Lookup table' approach to predicting an
appropriate score or verdict given a feature vector.

Assumes high levels of duplication in the feature vectors
with noisy targets.

This model aggregates verdicts or scores for identical
feature vectors. Each unique feature vectors is associated
with counts of associated verdicts or a list of scores.

Given a new feature vector, the table is used to predict the
appropriate respons as the mode of the associated verdicts for
classification or the average of the list of scores for regression.

If a feature vector is not in the lookup-table, the model
falls back to the global mode or mean for its predictions.
    """
    def __init__(self):
        self.default_prediction = None
        self.is_fitted_ = False
        self.lookup_table = None
        self.targets = None
        super().__init__()

    def fit(self, X, y):
        """Create a list of y's for identical X's.
Store this in a dictionary where the feature vector is the key,
and the value is a list of associated scores.
        ----------
        X : DataFrame, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        self.lookup_table = X.groupby(list(X.columns), sort=False).groups
        self.targets = y
        self.is_fitted_ = True
        return self


def _safemode(targets):
    """Returns the mode of a series, if multiple verdicts are equally
common, return the 'safest' (per domain knowledge) verdict."""
    safemodes = set(['-', 'n', 'neutral'])
    md = targets.mode()
    if len(md) == 1:
        return md[0]
    safe_md = list(safemodes.intersection(md))
    if safe_md:
        str(safe_md[0])
    return md[0]


class SDRLookupClassifier(_SDRLookupTable):
    """Returns the most frequent verdict of identical feature vectors
or the global mode of verdicts if the feature vector has not been observed.
    """
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        super().fit(X, y)
        self.default_prediction = _safemode(self.targets)
        return self

    def predict(self, X):
        """Average score for data seen in training, global average else."""
        check_is_fitted(self, 'is_fitted_')

        # Input validation
        X = check_array(X)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return [_safemode(self.targets[self.lookup_table[tuple(v)]])
                if tuple(v) in self.lookup_table
                else self.default_prediction
                for (_, v) in X.iterrows()]

    def score(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        preds = self.predict(X)
        return accuracy_score(y, preds)


class SDRLookupRegressor(_SDRLookupTable):
    """Returns the average score of identical feature vectors
or the global average if the feature vector has not been observed.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        super().fit(X, y)
        self.default_prediction = self.targets.mean()
        return self

    def predict(self, X):
        """Average score for data seen in training, global average else."""
        check_is_fitted(self, 'is_fitted_')

        # Input validation
        X = check_array(X)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return [self.targets[self.lookup_table[tuple(v)]].mean()
                if tuple(v) in self.lookup_table
                else self.default_prediction
                for (_, v) in X.iterrows()]

    def score(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        preds = self.predict(X)
        return r2_score(y, preds)


class SDRContextClassifier(BaseEstimator):
    """Make predictions on specific SDR contexts using the specified
'contexts' are specified as a dictionary of dictionaries:
contexts = {
    name: { # (Arbitrary) name for the context.
        'regex': r'column_name_pattern', # regular expression matched against
                                         # column names of dataframe X in the
                                         # fit, predict, and score functions.
        'c_idx': [1,2,3], # Column indices. Used to split out context data if
                          # no regex is provided of if X is not a dataframe.
        'model': clf, # optional, the model to be used for this context.
    } # , [more contexts]
}
If no 'model' is specified for a context, the 'baselearner'
    (a scikit compatible classifier) is used.
Returns the most penalizing per-context-verdict as per the
    highest score in 'verdict2priority'."""
    def __init__(self, contexts,
                       baselearner=SDRLookupClassifier,
                       baselearner_params=None,
                       scorer=accuracy_score,
                       verdict2priority={'+': 1,
                                         '?': 0,
                                         '-': -1}):
        self.baselearner = baselearner
        self.baselearner_params = baselearner_params
        self.contexts = contexts
        self.is_fitted_ = False
        self.models = {}
        self.scorer = scorer
        self.verdict2priority = verdict2priority
        self.priority2verdict = {v:k for k,v in verdict2priority.items()}
        super().__init__()

    @staticmethod
    def _get_context_data(X, context):
        if 'regex' in context and isinstance(X, pd.DataFrame):
            cX = X.filter(regex=context['regex'], axis=1)
            #print(f"{cX.shape, type(cX)}, {context['regex']}")
        else:
            raise ValueError('Only regex for pandas.dataframes '
                             'implemented at this point...')
        return cX
        
    def fit(self, X, y):
        # Allow model to do the verification, no need to duplicate effort.
        # _, y = check_X_y(X, y, accept_sparse=True)
        for name, context in self.contexts.items():
            if 'model' in context:
                clf = context['model']()
            elif self.baselearner:
                clf = self.baselearner()
            else:
                raise ValueError('No context model or baselearner set.')
            if 'model_params' in context:
                clf.set_params(**context['model_params'])
            elif self.baselearner_params:
                clf.set_params(**self.baselearner_params)
            cX = self._get_context_data(X, context)
            self.models[name] = clf
            self.models[name].fit(cX, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Most penalizing verdict from the per-context predictions."""
        check_is_fitted(self, 'is_fitted_')
        names = list(self.contexts.keys())
        cpreds = np.array(np.zeros((len(names), X.shape[0])), dtype='object')
        for cnt, name in enumerate(names):
            cX = self._get_context_data(X, self.contexts[name])
            cpreds[cnt] = self.models[name].predict(cX)
        cpreds = np.vectorize(self.verdict2priority.__getitem__)(cpreds)
        return np.vectorize(self.priority2verdict.__getitem__)(cpreds.max(axis=0))

    def score(self, X, y):
        check_is_fitted(self, 'is_fitted_')
        preds = self.predict(X)
        return self.scorer(y, preds)
