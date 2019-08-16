from genetic_algorithm.utils.sklearn_custom_transformers import TargetMeanEncoder

import logging
import numpy as np
import pandas as pd

# TEST -----

X = pd.DataFrame({'g1': ['a', 'a', 'b', 'b'], 'g2': ['c', 'd', 'd', 'd']})
y = pd.Series([1, 5, 7, 9])

tme = TargetMeanEncoder()
tme.fit(X, y)
assert (
        tme.transform(X) == np.array([
        [3, 1],
        [3, 7],
        [8, 7],
        [8, 7]
    ])
).all()

# TEST -----

tme = TargetMeanEncoder(priorFrac=0.5)
tme.fit(X, y)
assert (
    tme.transform(X) == np.array([
        [4.25, 4],
        [4.25, 6.4],
        [6.75, 6.4],
        [6.75, 6.4]
    ])
).all()

# TEST -----

tme = TargetMeanEncoder(priorSize=2)
tme.fit(X, y)
assert (
    tme.transform(X) == np.array([
        [4.25, 4],
        [4.25, 6.4],
        [6.75, 6.4],
        [6.75, 6.4]
    ])
).all()

logging.info('All tests passed')
