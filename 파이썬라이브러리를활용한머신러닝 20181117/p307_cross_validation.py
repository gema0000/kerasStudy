# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import mglearn
import pandas as pd
from sklearn.preprocessing import StandardScaler

# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)

################# matplotlib 한글 구현 #############################
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
####################################################################


from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# # 인위적인 데이터셋을 만듭니다
# X, y = make_blobs(random_state=0)
# # 데이터와 타깃 레이블을 훈련 세트와 테스트 세트로 나눕니다
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# # 모델 객체를 만들고 훈련 세트로 학습시킵니다
# logreg = LogisticRegression(solver='liblinear', multi_class='ovr').fit(X_train, y_train)
# # 모델을 테스트 세트로 평가합니다
# print("테스트 세트 점수: {:.2f}".format(logreg.score(X_test, y_test)))
# print(X.shape)
# print(y.shape)


# 교차검증 (분류: 계층별k-겹교차검증, 회규: k-겹교차검증)을 기본으로 사용.
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression(solver='liblinear', multi_class='ovr')

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)      # cv는 폴드의 수
print("교차 검증 점수: {}".format(scores))
print("교차 검증 평균 점수: {:.2f}".format(scores.mean()))
print(iris.data.shape)
print(iris.target.shape)


from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
scores2 = cross_val_score(logreg, cancer.data, cancer.target, cv=5)
print(scores2)


# 교차검증 KFold 사용
from sklearn.model_selection import cross_val_score, KFold
kfold = KFold(n_splits=5)
print(cross_val_score(logreg, iris.data, iris.target, cv=kfold))
kfold2 = KFold(n_splits=3)
print(cross_val_score(logreg, iris.data, iris.target, cv=kfold2))
########################################################################
kfold3 = KFold(n_splits=3, shuffle=True, random_state=0)
print(cross_val_score(logreg, iris.data, iris.target, cv=kfold3))
########################################################################


# # LOOCV  # 별로 안쓸만.
# from sklearn.model_selection import LeaveOneOut
# loo = LeaveOneOut()
# scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
# print("교차 검증 분할 횟수 : ", len(scores))
# print("평균 정확도 : {:.2f}".format(scores.mean()))

# 임의 분할 교차 검증 p.313
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("교차 검증 점수 : \n{}".format(scores))





