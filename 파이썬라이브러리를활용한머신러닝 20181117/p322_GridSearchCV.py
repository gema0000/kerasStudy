import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
iris = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

param_grid ={'C':[0.001, 0.01, 0.1, 1, 10, 100],
             'gamma' : [0.001, 0.01, 0.1, 1, 10, 100]}
print("매개변수 그리드 : \n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("테스트 세트 점수 : {:.2f}".format(grid_search.score(X_test, y_test)))

print("최적 매개변수 : {}".format(grid_search.best_params_))
print("최상 교차 검증 점수 : {:.2f}".format(grid_search.best_score_))   
        # best_score_ : 훈련세트에서 수행한 교차 검증의 평균 정확도가 저장

print("최고 성능 모델 : \n{}".format(grid_search.best_estimator_))

# 교차 검증 결과 분석
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)
print(results.head())

scores = np.array(results.mean_test_score).reshape(6,6)
print(scores)

