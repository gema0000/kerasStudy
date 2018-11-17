from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify = cancer.target, random_state=66
)

from sklearn.linear_model import Ridge


print("================== ridge ===================")
ridge = Ridge().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))

print("테스트 세트 점수 : {:.2f}".format(ridge.score(X_test, y_test)))


from sklearn.linear_model import Lasso      # L1 규제

print("================== Lasso  L1규제 ===================")
# lasso = Lasso().fit(X_train, y_train)                              # 기본 alpha = 0.1 
lasso = Lasso(alpha=0.01, max_iter=10000).fit(X_train, y_train)
print("훈련 세트 점수 : {:.2f}".format(lasso.score(X_train, y_train)))
print("테스트 세트 점수 : {:.2f}".format(lasso.score(X_test, y_test)))
print("사용한 특성의 수: {}".format(np.sum(lasso.coef_ !=0)))

