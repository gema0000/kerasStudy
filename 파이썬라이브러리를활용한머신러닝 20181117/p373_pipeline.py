# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import mglearn
import pandas as pd
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)

################# matplotlib 한글 구현 #############################
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
####################################################################

from sklearn.model_selection import GridSearchCV
param_grid = {'svm__C':[0.001, 0.01, 0.1, 1, 10, 100],
              'svm__gamma':[0.001, 0.01, 0.1, 1, 10, 100]}

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
# pipe.fit(X_train, y_train)

# print("테스트 점수 : {:.2f}".format(pipe.score(X_test, y_test)))

# aaa = SVC()
# aaa.fit(X_train, y_train)
# print(aaa.score(X_test, y_test))

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print("최상의 교차 검증 정확도 : {:.2f}".format(grid.best_score_))
print("테스트 세트 점수 : {:.2f}".format(grid.score(X_test, y_test)))
print("최적의 매개 변수 : {}".format(grid.best_params_))


