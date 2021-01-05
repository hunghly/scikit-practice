#!venv/bin/python
import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)
print(rng)
X = rng.rand(10,2000)
print(X)
X = np.array(X, dtype='float32')
print(X.dtype)
print(X)
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X) # casting float32 to float64
print(X_new.dtype)
print(X_new)