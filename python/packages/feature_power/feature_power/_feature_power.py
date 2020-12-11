# Feature Power implementation
from copy import deepcopy
from functools import lru_cache
from gmpy import fac


class FifoList:
    """Fifo, by courtesy of:
https://www.oreilly.com/library/view/python-cookbook/0596001673/ch17s15.html"""

    def __init__(self):
        self.data = {}
        self.nextin = 0
        self.nextout = 0

    def append(self, data):
        self.data[self.nextin] = data
        self.nextin += 1

    def pop(self):
        # Raises KeyError for empty Fifo.
        result = self.data[self.nextout]
        del self.data[self.nextout]
        self.nextout += 1
        return result

    def tolist(self):
        return [self.data[_id] for _id in range(self.nextout, self.nextin)
                if _id in self.data]


def _adjust_response(response,
                     discrimination_threshold,
                     f_0,
                     num_trees):
    return response - discrimination_threshold / num_trees - f_0


def _agg_coalition_space(jgbm,
                         method,
                         first_tree_default_prediction,
                         discrimination_threshold):
    if method not in ['pathPow', 'cumNodePow', 'strictNodePow']:
        raise ValueError('_agg_coalition_space: unknown method')
    # pathPow (root-to-leaf only):
    # coalition: list of internal nodes
    # value: abs of leaf
    # nodePow (root-to-internal):
    # coalition: list of internal nodes
    # value: 1
    trees = jgbm['tree_info']
    num_trees = len(trees)
    gbm_coalitions = []
    gbm_coalition_values = []

    for tree in trees:
        # Prepare data structures.
        subtree_fifo = FifoList()
        tree_number = int(tree['tree_index'])
        tree = deepcopy(tree['tree_structure'])
        f_0 = 0
        if tree_number == 0:
            f_0 = first_tree_default_prediction
        tree_coalitions = []
        tree_coalition_values = []
        while tree:
            coalition = tree.pop('path', [])
            left = tree.pop('left_child', None)
            right = tree.pop('right_child', None)
            internal_node = left and right
            if internal_node:
                split_var_id = tree['split_feature']
                coalition.append(split_var_id)
                if method in ['cumNodePow', 'strictNodePow']:
                    tree_coalitions.append(coalition)
                left['path'] = deepcopy(coalition)
                right['path'] = deepcopy(coalition)
                subtree_fifo.append(left)
                subtree_fifo.append(right)
            else:  # leaf node
                if method == 'pathPow':
                    coalition_value = abs(
                        _adjust_response(tree['leaf_value'],
                                         discrimination_threshold,
                                         f_0,
                                         num_trees))
                    tree_coalitions.append(coalition)
                    tree_coalition_values.append(coalition_value)
            try:
                tree = subtree_fifo.pop()
            except KeyError:
                tree = None

            gbm_coalitions.extend(tree_coalitions)
            if method == 'pathPow':
                gbm_coalition_values.extend(tree_coalition_values)

    # Combine coalitions and values of characteristic functions (summed over all leaves).
    if method == 'pathPow':
        return [(c, v) for c, v in zip(gbm_coalitions, gbm_coalition_values)]
    else:  # nodePow
        return [(c, 1) for c in gbm_coalitions]


@lru_cache(1000)  # set to max_tree_depth for full caching
def _coalition_quotient_numerator(coalition_size, num_features, inclusion):
    if inclusion:
        numerator = _lru_fac(coalition_size - 1) * _lru_fac(num_features - coalition_size)
    else:
        numerator = _lru_fac(coalition_size) * _lru_fac(num_features - coalition_size - 1)
    return numerator


@lru_cache(1000)  # set to num_features for full caching
def _lru_fac(x):
    return fac(x)


def _has_same_sign(x, y):
    if x == y == 0:
        return True
    return x * y > 0


def _split_coalition_space(method, coalition_space, feature_id):
    # pathPow
    # In paper: p_pp = [n0,...,ni, leaf_val]
    #           s = d(p_pp) # edges
    # In code:  p_pp = (p=[no,...ni], leaf_val)
    #           s = len(p)
    #
    # nodePow
    # In paper: p_np = [n0,...,ni]
    #           s = d(p_pp)+1 # edges
    # In code:  p_np = (p=[no,...ni], func_over_T_ni)
    #           s = len(p)
    if method == 'pathPow':
        # inclusion: feature in path?
        # inc,exc: s = length of path
        coalition_space_inc = [(len(coalition), value)
                               for coalition, value in coalition_space
                               if feature_id in coalition]
        coalition_space_exc = [(len(coalition), value)
                               for coalition, value in coalition_space
                               if feature_id not in coalition]
    elif method == 'cumNodePow':
        # inclusion: feature in path?
        # inc: s = length up to and including first mention of feature (index + 1)
        # exc: s = length of path
        coalition_space_inc = [(coalition.index(feature_id) + 1, value)
                               for coalition, value in coalition_space
                               if feature_id in coalition]
        coalition_space_exc = [(len(coalition), value)
                               for coalition, value in coalition_space
                               if feature_id not in coalition]
    elif method == 'strictNodePow':
        # inclusion: path ends in feature?
        # inc: s = length up to and including first mention of feature (index + 1)
        # exc: s = length of path
        coalition_space_inc = [(coalition.index(feature_id) + 1, value)
                               for coalition, value in coalition_space
                               if coalition[-1] == feature_id]
        coalition_space_exc = [(len(coalition), value)
                               for coalition, value in coalition_space
                               if coalition[-1] != feature_id]
    else:
        raise ValueError('_coalition_space: unknown method')

    return coalition_space_inc, coalition_space_exc


def regrGBM_FP_agg(jgbm,  # json representation of the model
                   method,  # pathPow, cumNodePow, strictNodePow
                   first_tree_default_prediction,  # f_0
                   discrimination_threshold):  # \rho

    if method not in ['pathPow', 'cumNodePow', 'strictNodePow']:
        raise ValueError('regrGBM_FP_agg: unknown method')

    # Compute coalition space and values of characteristic functions
    # according to "method":
    coalition_space = _agg_coalition_space(jgbm, method,
                             first_tree_default_prediction,
                             discrimination_threshold)
    k = len(coalition_space)

    # Determine Feature Power aggregate for each feature:
    num_features = len(jgbm['feature_names'])
    fp_agg = [0] * num_features
    coalition_quotient_denominator = _lru_fac(num_features)
    for feature_id in range(num_features):
        # Split coalitions in those including and excluding current feature:
        coalition_space_inc, coalition_space_exc = _split_coalition_space(
            method, coalition_space, feature_id)

        # Compute sums over coalitions:
        inc_sum = sum([_coalition_quotient_numerator(coalition_size,
                                                     num_features,
                                                     True) * value
                       for coalition_size, value in coalition_space_inc])
        exc_sum = sum([_coalition_quotient_numerator(coalition_size,
                                                     num_features,
                                                     False) * value
                       for coalition_size, value in coalition_space_exc])
        # Combine into FP by dividing by K:
        fp_agg[feature_id] = float((inc_sum - exc_sum) /
                                   (k * coalition_quotient_denominator))

    # Return vector of aggregate feature power:
    return fp_agg
