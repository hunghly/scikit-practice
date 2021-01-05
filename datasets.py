#!venv/bin/python
#%%
"""
Scikit-learn deals with learning information from one or 
more datasets that are represented as 2D arrays. 
They can be understood as a list of multi-dimensional 
observations. We say that the first axis of these arrays is 
the samples axis, while the second is the features axis.
"""
from sklearn import datasets
# import matplotlib.pyplot as plt 
from matplotlib import pyplot as plt

iris = datasets.load_iris()
data = iris.data
print(data.shape) # describes the shape of the data set
# (150, 4) which means 150 samples(observations) and 4 features
#When the data is not initially in the (n_samples, n_features) shape, it \
# needs to be preprocessed in order to be used by scikit-learn.

digits = datasets.load_digits()
digits.images.shape
(1797, 8, 8)
# print(digits.images[1])
plt.imshow(digits.images[-1],
            cmap=plt.cm.gray_r)

data = digits.images.reshape((digits.images.shape[0], -1)
)
print(data)
# plt.imshow(digits.images[2],
#             cmap=plt.cm.gray_r)
# %%
