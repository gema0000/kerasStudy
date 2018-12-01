# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import mglearn
import pandas as pd
from sklearn.preprocessing import StandardScaler

################# matplotlib 한글 구현 #############################
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
####################################################################

cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)

import sklearn
print(sklearn.__version__)      # 0.19.1

from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 인위적인 데이터셋을 만듭니다
X, y = make_blobs(random_state=0)
# 데이터와 타깃 레이블을 훈련 세트와 테스트 세트로 나눕니다
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 모델 객체를 만들고 훈련 세트로 학습시킵니다
logreg = LogisticRegression(solver='liblinear', multi_class='ovr').fit(X_train, y_train)
# 모델을 테스트 세트로 평가합니다
print("테스트 세트 점수: {:.2f}".format(logreg.score(X_test, y_test)))
print(X.shape)
print(y.shape)
print(X)
print(y)
print(type(X))
