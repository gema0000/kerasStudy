
import numpy as np
from scipy import sparse

eye = np.eye(4)

print(eye)

sparse_matrix = sparse.csr_matrix(eye)
print(sparse_matrix)

# data = np.ones(4)

import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = np.sin(x)

# plt.plot(x,y,marker='p')
# plt.show()

from IPython.display import display
import pandas as pd

# 회원 정보가 들어간 간단한 데이터셋을 생성합니다.
data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location' : ["New York", "Paris", "Berlin", "London"],
        'Age' : [24, 13, 53, 33]
       }

data_pandas = pd.DataFrame(data)
# IPython.display는 주피터 노트북에서 Dataframe을 미려하게 출력해줍니다.
display(data_pandas)

import scipy as sp
print("Scipy 버전 {}".format(sp.__version__))

from sklearn.datasets import load_iris
iris_dataset = load_iris()

print(iris_dataset.keys())

# print(iris_dataset)

print(type(iris_dataset))
print(iris_dataset['feature_names'])
print(type(iris_dataset['feature_names']))
print(iris_dataset['data'].shape)

from sklearn.model_selection import train_test_split    # 기본 25%
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

import mglearn

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o',
                           hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3 )

# plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)       # 이거 실행하면 아래처럼 나온다.
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#                      metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')

prediction =knn.predict(X_test)

print(prediction)

acc = knn.score(X_test, y_test)

print(acc)


