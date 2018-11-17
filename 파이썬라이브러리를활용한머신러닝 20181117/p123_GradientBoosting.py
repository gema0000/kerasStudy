from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import mglearn
import pandas as pd

################# matplotlib 한글 구현 #############################
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
####################################################################

from sklearn.ensemble import GradientBoostingClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)

print(cancer.data.shape)        # 569, 30
print(cancer.target.shape)      # 569,

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("훈련 세트 점수: {:.2f}".format(gbrt.score(X_train, y_train)))    # 1.0
print("테스트 세트 점수 : {:.2f}".format(gbrt.score(X_test, y_test)))   # 0.96

########### 과대적합이어서 트리의 최대깊이를 줄여 사전 가지치기를 강하게 하거나 학습률을 낮출수 있다.

# 01.
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

print("훈련 세트 점수: {:.2f}".format(gbrt.score(X_train, y_train)))    # 0.99
print("테스트 세트 점수 : {:.2f}".format(gbrt.score(X_test, y_test)))   # 0.97

# 02.
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)

print("훈련 세트 점수: {:.2f}".format(gbrt.score(X_train, y_train)))    # 0.99
print("테스트 세트 점수 : {:.2f}".format(gbrt.score(X_test, y_test)))   # 0.96

