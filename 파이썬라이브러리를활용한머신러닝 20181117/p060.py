from sklearn.datasets import load_breast_cancer
import numpy as np
import mglearn

canser = load_breast_cancer()

# print(canser)
print(canser.keys())
print(canser.target_names)
print(canser.feature_names)
print(canser.data.shape)        # 569, 30
print(canser.target.shape)      # 569,
print("================================")
print(type(canser))
print(type(canser.data))
print(type(canser.target))
print(type(canser.target_names))
print(type(canser.feature_names))
print(type(canser.DESCR))
print(type(canser.data))

# print(np.bincount(cancer.target))

from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)    # 506, 13

X, y = mglearn.datasets.load_extended_boston()
print(X.shape)              # 506, 104

from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("======================================")
print(X)
print(y)
