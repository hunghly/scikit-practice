#!venv/bin/python
#%%
from sklearn import datasets, svm
import numpy as np

"""
scikit-learn comes with a few standard datasets, for instance the 
-iris and digits datasets for classification and 
-the diabetes dataset for regression.
"""
iris = datasets.load_iris()
digits = datasets.load_digits()

# print(iris.data)
#print("target", digits.target) # the numeber corresponding to each digit image that we are trying to learn
#print('---------')
#print(digits.data) # gives access to the features that can be used to classify the digits samples
#print('---------')
print(digits.keys())
# print(digits.DESCR)
# print(len(digits.feature_names))
print(digits.target)
print(digits.data.shape)
# print(np.unique(digits.target))
# print(digits.target_names)
# print(digits.data[0])
print(digits.images.shape)
"""
You can visually check that the images and the data are related by reshaping the images array to two dimensions: digits.images.reshape((1797, 64)).
"""

"""
print(digits.data)
print('=================')
print(digits.images.reshape((1797,64)))
print('=================')
print(digits.images)
print('=================')
print(digits.data.reshape((1797, 8, 8)))
# print(np.all(digits.images.reshape(1797, 64)) == digits.data))
# print(np.all(digits.images.reshape((1797,64))))
print(np.all(digits.images.reshape((1797,64)) == digits.data))
"""

"""
the data is always a 2D array, shape (n_samples, n_features), although the original data may have had a different shape. In the case of the digits, each original sample is an image of shape (8, 8) and can be accessed using: digits.images
"""
# print(digits.images[0])

classifier = svm.SVC(gamma=0.001, C=100) #setting the classifier/estimator instance
print(str(classifier.fit(digits.data[:-1], digits.target[:-1])))
print(str(classifier.predict(digits.data[-1:])))
# print(classifier.fit(digits.data[:-1], digits[:-1]))

"""
# Import matplotlib
import matplotlib.pyplot as plt

# Figure size (width, height) in inches
fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

# Show the plot
plt.show()
"""
# Import matplotlib
import matplotlib.pyplot as plt 

# Join the images and target labels in a list
images_and_labels = list(zip(digits.images, digits.target))
print(images_and_labels)

# for every element in the list
for index, (image, label) in enumerate(images_and_labels[:8]):
    # initialize a subplot of 2X4 at the i+1-th position
    plt.subplot(2, 4, index + 1)
    # Don't plot any axes
    plt.axis('off')
    # Display images in all subplots 
    plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
    # Add a title to each subplot
    plt.title('Training: ' + str(label))

# Show the plot
plt.show()
# %%
