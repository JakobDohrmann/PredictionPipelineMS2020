import pytest

from sklearn.utils.estimator_checks import check_estimator

from sdrmodels import SDRLookupRegressor
from sdrmodels import SDRLookupClassifier


@pytest.mark.parametrize(
    "Estimator", [SDRLookupRegressor, SDRLookupClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
